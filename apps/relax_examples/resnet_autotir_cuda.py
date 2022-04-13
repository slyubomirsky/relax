# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Tuning Resnet with AutoTIR."""

import tvm
import tvm.testing
from tvm.relay import testing
from tvm import relax, relay
from tvm.relax.testing import relay_translator, nn
from tvm.runtime import vm as vm_rt
import numpy as np
import os
import tempfile
import time
from tvm.meta_schedule.apply_history_best import ApplyHistoryBest
from tvm.meta_schedule.utils import autotvm_silencer
from tvm.meta_schedule.database import JSONDatabase
from tvm import transform
from tvm.ir.transform import PassContext
from tvm.contrib import graph_executor
from tvm import meta_schedule as ms
from tvm.relay import build as relay_build
from tvm.meta_schedule.builder import LocalBuilder
from tvm.meta_schedule.runner import LocalRunner


def get_resnet(batch_size, dtype, layout, image_shape):
    relay_mod, params = testing.resnet.get_workload(
        num_layers=50,
        batch_size=batch_size,
        dtype=dtype,
        layout=layout,
        image_shape=image_shape,
    )
    relax_mod = relay_translator.from_relay(relay_mod["main"])
    return relay_mod, params, relax_mod


def create_database(network, layout, batch_size, target, is_relax=True):
    workload_file = "%s_autotir_logs/%s-%s-B%d-%s-workload.json" % (
        "relax" if is_relax else "relay",
        network,
        layout,
        batch_size,
        target.kind.name,
    )

    tuning_record_file = "%s_autotir_logs/%s-%s-B%d-%s-tuning_record.json" % (
        "relax" if is_relax else "relay",
        network,
        layout,
        batch_size,
        target.kind.name,
    )
    is_tuned = os.path.exists(workload_file)
    os.makedirs(os.path.dirname(workload_file), exist_ok=True)

    database = JSONDatabase(
        path_workload=workload_file,
        path_tuning_record=tuning_record_file,
    )
    return database, is_tuned


if __name__ == "__main__":
    target = tvm.target.Target("nvidia/nvidia-v100")
    # target = tvm.target.Target("nvidia/geforce-rtx-2080-ti") for Kraken
    network = "resnet-50"
    batch_size = 1
    layout = "NCHW"
    dtype = "float32"
    # layout = "NHWC"
    if layout == "NHWC":
        image_shape = (224, 224, 3)
    elif layout == "NCHW":
        image_shape = (3, 224, 224)
    else:
        raise ValueError("Invalid layout: " + layout)
    input_shape = (batch_size,) + image_shape
    total_trials = 20000
    times = 1000
    input_name = "data"

    relay_database, relay_tuned = create_database(
        network, layout, batch_size, target, is_relax=False
    )
    relax_database, relax_tuned = create_database(network, layout, batch_size, target)

    relay_mod, relay_params, relax_mod = get_resnet(batch_size, dtype, layout, image_shape)

    if relay_tuned:
        with target, autotvm_silencer(), ApplyHistoryBest(relay_database):
            with PassContext(
                opt_level=3,
                config={"relay.backend.use_meta_schedule": True},
            ):
                relay_ex = relay_build(relay_mod, target=target, params=relay_params)
    else:
        # Tune Relay resnet
        with tempfile.TemporaryDirectory() as work_dir:
            relay_ex = ms.tune_relay(
                mod=relay_mod,
                params=relay_params,
                target=target,
                config=ms.EvolutionarySearchConfig(
                    num_trials_per_iter=32,  # 32 / 64
                    max_trials_per_task=total_trials,  # 20000
                    max_trials_global=total_trials,  # 20000
                ),
                work_dir=work_dir,
                database=relay_database,
            )

    input_shape = (1, 3, 224, 224)
    dev = tvm.cpu() if str(target).startswith("llvm") else tvm.cuda()
    data = tvm.nd.array(np.random.rand(*input_shape).astype(np.float32), dev)
    params = nn.init_params(relax_mod)

    # measure relay performance
    module = graph_executor.GraphModule(relay_ex["default"](dev))
    module.set_input(input_name, data)
    # warmup
    module.run()
    # get the output
    # module.get_output(0).numpy()
    tic = time.time()
    for i in range(times):
        module.run()
    toc = time.time()
    e0 = (toc - tic) * 1000 / times

    print(f"relay resnet50 inference perf: {e0} ms")

    if relax_tuned:
        with transform.PassContext(opt_level=3):
            relax_mod = relax.transform.MetaScheduleApplyHistoryBest(relax_database, target)(
                relax_mod
            )
            relax_ex = relax.vm.build(relax_mod, target=target)
    else:
        # Tune Relax resnet
        with tempfile.TemporaryDirectory() as work_dir:
            relax_ex = ms.tune_relax(
                mod=relax_mod,
                target=target,
                config=ms.EvolutionarySearchConfig(
                    num_trials_per_iter=32,  # 32 / 64
                    max_trials_per_task=total_trials,  # 20000
                    max_trials_global=total_trials,  # 20000
                ),
                work_dir=work_dir,
                database=relax_database,
            )

    # measure relax performance
    relax_vm = relax.VirtualMachine(relax_ex, dev)
    # warmup
    res = relax_vm["main"](data, *params)
    tic = time.time()
    for i in range(times):
        relax_vm["main"](data, *params)
    toc = time.time()
    e1 = (toc - tic) * 1000 / times

    print(f"relax resnet50 inference perf: {e1} ms")
