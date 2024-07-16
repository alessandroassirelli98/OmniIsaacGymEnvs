from typing import Optional

from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims import GeometryPrimView, RigidPrimView, XFormPrimView
import torch


class DrillView(ArticulationView):
    def __init__(
        self,
        prim_paths_expr: str,
        name: Optional[str] = "DrillView",

    ) -> None:
        """[summary]"""

        super().__init__(prim_paths_expr=prim_paths_expr, name=name, reset_xform_properties=False)


        # self._index_targets = XFormPrimView(prim_paths_expr="/World/envs/.*/drill/index_target", name="index_target_view")
        self._index_fingertip_targets = XFormPrimView(prim_paths_expr="/World/envs/.*/drill/index_fingertip_target", name="index_fingertiptarget_view")
        # self._middle_targets = XFormPrimView(prim_paths_expr="/World/envs/.*/drill/middle_target", name="middle_ringt_view")
        # self._ring_targets = XFormPrimView(prim_paths_expr="/World/envs/.*/drill/ring_target", name="ring_target_view")
        # self._little_targets = XFormPrimView(prim_paths_expr="/World/envs/.*/drill/little_target", name="little_target_view")
        # self._thumb_targets = XFormPrimView(prim_paths_expr="/World/envs/.*/drill/thumb_target", name="thumb_target_view")