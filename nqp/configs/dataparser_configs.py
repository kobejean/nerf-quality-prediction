"""
Nerfstudio Template Config

Define your custom method here that registers with Nerfstudio CLI.
"""

from __future__ import annotations

from nerfstudio.plugins.registry_dataparser import DataParserSpecification
from nqp.data.dataparsers.nqp_dataparser import NQPDataParserConfig
from nqp.data.dataparsers.nqp_blender_dataparser import NQPBlenderDataParserConfig
from nqp.data.dataparsers.nqp_minimal_dataparser import NQPMinimalDataParserConfig

nqp_dataparser = DataParserSpecification(config=NQPDataParserConfig())
nqp_blender_dataparser = DataParserSpecification(config=NQPBlenderDataParserConfig())
nqp_minimal_dataparser = DataParserSpecification(config=NQPMinimalDataParserConfig())
