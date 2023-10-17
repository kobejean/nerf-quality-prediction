"""
Nerfstudio Template Config

Define your custom method here that registers with Nerfstudio CLI.
"""

from __future__ import annotations

from nerfstudio.plugins.registry_dataparser import DataParserSpecification
from nqp.data.dataparsers.nqp_dataparser import NQPDataParserConfig

nqp_dataparser = DataParserSpecification(config=NQPDataParserConfig())
