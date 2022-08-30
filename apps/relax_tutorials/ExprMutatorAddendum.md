# Further Notes on Using `ExprMutator`

The previous pass-writing tutorial discussed how `ExprMutator` can be used to implement program transformations by traversing an AST and replacing subtrees with newly constructed ones that implement some new logic or optimization. The tutorial described the overall logic for these AST traversals and the functionality of `ExprMutator` with a detailed example, but the `ExprMutator` implementation itself contains further APIs for processing programs and additionally maintains some internal state that custom passes must be aware of. This document will briefly survey some of these utility methods and discuss how custom passes can interact with the `ExprMutator`'s internal state.

## Table of Contents

1. [`BlockBuilder` and Its Utilities](#blockbuilder-and-its-utilities)
    1. [Constructing Bindings](#constructing-bindings)
    2. [Renormalization](#renormalization)
    3. [Managing Global Definitions](#managing-global-definitions)
    4. [Reasoning about Shape Expressions](#reasoning-about-shape-expressions)
2. [Processing Bindings in `ExprFunctor`](#processing-bindings-in-exprfunctor)
    1. [Handling Scope](#handling-scope)
    2. [Processing Binding Blocks Using `BlockBuilder`](#processing-binding-blocks-using-blockbuilder)
    3. [Variable Remapping](#variable-remapping)
3. [Post-Order Traversal](#post-order-traversal)
4. [Python APIs](#python-apis)

## `BlockBuilder` and Its Utilities

An `ExprMutator` keeps an internal `BlockBuilder` called `builder_` to assist with processing `SeqExpr`s and (per the name) constructing `BindingBlock`s. The `BlockBuilder` provides many helper functions that track states related to the current binding block and also generally provides a convenient API for accumulating bindings and producing a binding block.

### Constructing Bindings

`BlockBuilder` allows for constructing binding blocks in a procedural manner, adding in one binding at a time and returning the complete binding block once it is finished rather than constructing all of the bindings in advance and then creating the binding block.

For example, consider the following way to define a binding block explicitly:

```cpp
// must define all the variables in advance in case the expressions use them
Var v1 = Var("v1");
Var v2 = Var("v2");
// ...
Var vk = Var("vk");
// ...
Var vn = Var("vn");
BindingBlock b = BindingBlock({
  VarBinding(v1, e1),
  VarBinding(v2, e2),
  ...,
  MatchShape(ek, shape_pattern, vk),
  ...,
  VarBinding(vn, en)
});
```

With the `BlockBuilder` (let's call it `builder_`), it would resemble the following:

```cpp
builder_.BeginBindingBlock();
Var v1 = builder_.Emit(e1, "v1");
Var v2 = builder_.Emit(e2, "v2");
// ...
Var vk = builder_.EmitMatchShape(ek, shape_pattern, "vk");
// ...
Var vn = builder_.Emit(en, "vn");
BindingBlock b = builder_.EndBlock();
```

The `Emit` method creates a new variable, adds a binding of the argument expression to that variable, and returns the variable so that it could be easily referenced in the later expressions in that binding block. Another advantage of this approach is that a binding could easily be inserted into the middle of the block without having to add more declarations before the block like in the first example.

`EmitMatchShape` behaves similarly to `Emit` but inserts a `MatchShape` binding instead of an ordinary `VarBinding`. The builder also tracks whether the current block is an ordinary `BindingBlock` or a `DataflowBlock`. In the latter case, `Emit` returns a `DataflowVar` by default and `EmitOutput` must be used to produce a `Var` that leaves the `DataflowBlock`'s scope.

### Renormalization

As mentioned in the previous tutorial, `ExprMutator` relies on the assumption that the input programs will be in A-normal form (ANF). It would be inconvenient for pass implementations to manually maintain the ANF as they potentially change the AST, so `BlockBuilder` includes a `Normalize` method that automatically converts a given Relax expression into ANF. Notably, if an expression within a binding block is not in ANF, the normalization will turn the subexpressions into new bindings in that same block.

Renormalization also serves another purpose: Updating the checked types (`checked_type_`) and shape computation fields (`shape_`) of expressions by performing type and shape inference again. A program transformation may affect the inferred types or shapes and it would be troublesome and error-prone for pass authors to maintain the correctness of `checked_type_` and `shape_` manually. Instead, calling `BlockBuilder::Normalize` will do so automatically.

*Note that `ExprMutator` will renormalize every expression it returns after performing the recursive visit.* This allows for ANF passes to be easily composed, since it maintains the invariant that the output will also be in ANF.

```cpp
Expr ExprMutator::VisitExpr(const Expr& expr) {
  return builder_->Normalize(ExprFunctor::VisitExpr(expr));
}
```

### Managing Global Definitions

`BlockBuilder` also keeps track of the overall `IRModule` and provides convenience methods for retrieving the `IRModule` (`GetContextIRModule`), adding new functions to the `IRModule` (`AddFunction`) or replacing a definition with another one (`UpdateFunction`). `ExprMutator` constructors have an optional module argument. If the `IRModule` is provided, these convenience methods can be used to manage it from within the `ExprMutator`; it will have copy-on-write semantics. Managing the `IRModule` through `BlockBuilder` is useful for passes like lambda-lifting (`src/relax/transform/lambda_lift.cc`), which turns local function definitions in Relax programs into global ones and makes use of `BlockBuilder::UpdateFunction` and direct modifications to the `IRModule`.

### Reasoning about Shape Expressions

`BlockBuilder` provides a helper method for determining if two shape expressions are equivalent, `CanProveShapeEqual`. This is useful for determining if a change to the program has affected the given shape computation in a nontrivial way. Since shape expressions can contain arithmetic via `PrimExpr`, truly determining if shapes are equivalent would require algebraic manipulations and more advanced techniques. This method is conservative: It will never falsely say that two shapes can be proven equivalent, but it may miss two shapes that are, in fact, equal. In particular, it does not reason about what shape variables are in scope and what they may be bound to. It uses TVM's arithmetic analyzer (`src/arith/analyzer.cc`) to handle the proving.

## Processing Bindings in `ExprMutator`

Given the APIs and conveniences afforded by `BlockBuilder`, `ExprMutator` uses a stateful workflow for traversing binding blocks and processing variable definitions. Though this requires users to reason about the current state of the `BlockBuilder`, it allows for easily inserting bindings using `BlockBuilder::Emit` and its variants much more concisely compared to an approach that explicitly constructs the entire binding block all at once.

It is important for passes to correctly interact with this internal state in order to avoid introducing malformed constructs or creating other inconsistencies in the AST.

### Handling Scope

In order to allow for adding new local bindings from anywhere in the program, `ExprMutator` uses a helper method `ExprMutator::VisitWithNewScope(const Expr&)`. This method creates a new binding block via `BlockBuilder::BeginBindingBlock()` and then visits the given expression. During the visit to the argument expression, calls to `BlockBuilder::Emit` and its variants will introduce new bindings into the current binding block. Once the visit is complete, the binding block is wrapped in a `SeqExpr` whose `body` value is the final returned expression after visiting (or simply the `body` if the final binding block contains no bindings).

`VisitWithNewScope` is used by `ExprMutator` for visiting the body of a function definition and when visiting the true and false branches of an `If` node. Thus, an impliciation of `VisitWithNewScope`, is that, other than in the initial visit to a global function in the `IRModule`, the expression being returned is being preceded by a sequence of bindings and calling `BlockBuilder::Emit` will introduce one more binding to it.

### Processing Binding Blocks Using `BlockBuilder`

When the `ExprMutator` encounters a `SeqExpr`, it will visit the binding blocks in sequence and, within each block, it will visit the bindings in sequence. This will affect the state of the internal `BlockBuilder`.

Namely, the visit proceeds as follows:

```cpp
BindingBlock ExprMutator::VisitBindingBlock_(const BindingBlockNode* block) {
  builder_->BeginBindingBlock();
  for (Binding binding : block->bindings) {
    this->VisitBinding(binding);
  }
  return builder_->EndBlock();
}

// and analogously for dataflow blocks
```

Thus, visits to the individual bindings can use `BlockBuilder::Emit` to insert bindings into the current block (indeed, `VisitBinding` has a `void` return type, so `Emit` *must* be used to insert bindings into the block). The individual visits can call `Emit` multiple times to add multiple bindings into the current block; the visits do not need to map bindings one-to-one.

### Variable Remapping

Modifications to the AST may result in some variables' inferred types and shapes changing. This can lead to potential AST inconsistencies if the `checked_type_` and `shape_` fields of variables are out of date (i.e., if the updated variable is used in some places but an older version is used in others). Internally, Relax uses pointer identity to keep track of variables, so having distinct variables with different `checked_type_` and shape fields will be treated as a well-formedness error (only one would have a binding). To avoid this situation, `ExprMutator` keeps an internal mapping called `var_remap_`, which maps variable IDs to the most up-to-date `Var` node.

When a binding is visited, the `ExprMutator` uses the `WithShapeAndType` field to create a var node after visiting the bound expression. `WithShapeAndType` will use the transformed expression's inferred shape and type to create the new var node and compare it to the old one. If the new shape and type differ from the original, `var_remap_` will be updated; otherwise, the old variable will continue to be used. If the variable has been replaced, the `ExprMutator` will replace the uses of old var node with the new one via the variable map:

```cpp
Expr ExprMutator::VisitExpr_(const VarNode* op) {
  auto it = var_remap_.find(op->vid);
  if (it != var_remap_.end()) {
    return it->second;
  }

  return GetRef<Expr>(op);
}

// and analogously for DataflowVarNode
```

Note that there is further special handling for variable definitions. Instead of `VisitExpr_(const VarNode*)`, the `ExprMutator` calls `VisitVarDef_` when visiting bindings. `VisitVarDef_` will visit the associated shape computation (`shape_`) for the variable definition, performing any necessary transformations there as well. If this visit results in a change to the `shape_` field, it will update the variable map. It is important to visit the `shape_` field because it can contain variables from elsewhere in the program that may need to be remapped as well.

```cpp
Var ExprMutator::VisitVarDef_(const VarNode* var) {
  bool shape_unchanged = true;
  Expr new_shape;
  if (var->shape_) {
    new_shape = this->VisitExpr(Downcast<Expr>(var->shape_.value()));
    shape_unchanged &= new_shape.same_as(var->shape_);
  }

  if (shape_unchanged) {
    return GetRef<Var>(var);
  } else {
    Var new_var = DataflowVar(var->vid, NullOpt, var->checked_type_, var->span);
    UpdateShape(new_var, new_shape);

    this->var_remap_[var->vid] = new_var;
    return new_var;
  }
}

// and analogously for DataflowVarNodes
```

The default workflow for visiting a binding proceeds a follows:

1. First it visits the bound value, performing any necessary transformation.
2. Next it visits the newly defined variable via `VisitVarDef_`, updating the `shape_` field.
3. If the variable after the visit to `VisitVarDef_` has a differnt `checked_type_` and `shape_` from the original, update the variable mapping.
4. Emit the new binding (if it is different from the original).

Note that visits to `MatchShape` will also visit the shape pattern (by wrapping it in a `ShapeExpr` and using `VisitExpr` on it). Overrides to `VisitExpr_(const ShapeExpr&)` should take that into account as well.

## Post-Order Traversal

As discussed in the previous tutorial, certain passes can be implemented easily by applying functions to a post-order AST traversal using `PostOrderVisit`. These post-order visits process a given expression by first visiting the child subexpressions and then applying a function. `ExprMutator` has a similar convenience method for visiting an expression’s child subexpressions in order, called `VisitPostOrder_`. This method visits the expression and then normalizes the result, as shown below. Calling `VisitPostOrder_` at the beginning of a `VisitExpr_` override can be used to implement a post-order transformation; additionally, the method can be overridden to apply more transformations after the visit.

```cpp
template <typename T>
Expr VisitExprPostOrder_(const T* op) {
  return builder_->Normalize(ExprMutator::VisitExpr_(op));
}
```

For example, `tests/python/relax/test_expr_functor.py` uses the post-order visit to implement a post-order AST printer:

```python
@relax.expr_functor.mutator
class ASTPostPrinterMutator(PyExprMutator):
    """Print relax AST in the post order format."""

    def __init__(self) -> None:
        super().__init__()
        self.log = ASTLog()

    def visit_constant_(self, op: Constant) -> Expr:
        op = self.visit_expr_post_order(op)
        self.log.add("Constant")
        return op

    # ...

    def visit_tuple_(self, op: Tuple) -> Expr:
        op = self.visit_expr_post_order(op)
        self.log.add("Tuple")
        return op
```

This mutator will rebuild the AST it is given but also produce a log of the constructs encountered in the post-order traversal.

## Python APIs

The examples in this file have used C++ extensively. However, the APIs of `ExprVisitor` can be accessed from Python using the parent class `PyExprMutator` with the `@relax.expr_functor.mutator` decorator, like in the preceding section. The decorator is used to report any Python overrides over the FFI, ensuring that the recursive visits will call the user’s overrides of the original methods. If the decorator is omitted, then the FFI visits will not know about methods that were overridden in Python, so it is almost always desirable to include the decorator if implementing a mutator in Python. (The same is analogously true of visitors, which have a `PyExprVisitor` counterpart.)

The `PyExprMutator` class has analogous methods, like `with_shape_and_type`, `visit_with_new_scope`, and `get_var_remap` that call into their C++ implementations over the FFI. Similarly, methods like `visit_binding_block`, `visit_var_def`, `visit_expr_post_order`, and `visit_binding` will also all call the C++ implementations via the FFI. Additionally, `PyExprMutator` also has access (via the FFI) to the internal `BlockBuilder` via its `self.builder` field.

Note that the Python `BlockBuilder` has a slightly different interface from the C++ one:

- For constructing `DataflowBlock`s in Python, it provides a method `dataflow_block` that is meant to be used in `with` blocks:
    
    ```python
    with bb.dataflow_block():
        bb.emit(...)
        ...
    ```
    
- Similarly, it provides an API for constructing functions that also works through `with` blocks. The final returned value should be given using `emit_func_output`. The function will be added into the `IRModule`.
    
    ```python
    with bb.function("func_name", [arg1, arg2, ..., argn]):
        bb.emit(...)
        ...
        bb.emit_func_output(ret)
    ```
    
- There is a further convenience method for inserting calls to functions defined in TE, `emit_te`. It will insert the given TE function into the current `IRModule` as a TIR `PrimFunc` and insert a corresponding call to Relax’s `call_tir` operator.
