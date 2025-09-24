from datetime import timedelta
from kaapana.operators.KaapanaBaseOperator import KaapanaBaseOperator
from kaapana.blueprints.kaapana_global_variables import DEFAULT_REGISTRY, KAAPANA_BUILD_VERSION


class MyIvimFittingOperator(KaapanaBaseOperator):
    def __init__(
            self,
            dag,
            name="ivim-fitting-method",
            execution_timeout=timedelta(hours=5),
            ram_mem_mb=4000,
            ram_mem_mb_lmt=20000,
            *args,
            **kwargs,
            ):
        super().__init__(
                dag=dag,
                name=name,
                image=f"{DEFAULT_REGISTRY}/ivim-fitting-method:{KAAPANA_BUILD_VERSION}",
                image_pull_secrets=["registry-secret"],
                execution_timeout=execution_timeout,
                ram_mem_mb=ram_mem_mb,
                ram_mem_mb_lmt=ram_mem_mb_lmt,
                *args,
                **kwargs,
                )
