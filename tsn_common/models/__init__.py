"""
SQLAlchemy database models for TSN V2.
"""

from tsn_common.models.base import Base, GUID
from tsn_common.models.audio import AudioFile, AudioFileState
from tsn_common.models.transcription import Transcription, TranscriptionBackend
from tsn_common.models.callsign import Callsign, CallsignLog, CallsignTopic, ValidationMethod
from tsn_common.models.net import NetControlSession, NetSession, NetParticipation, CheckinType
from tsn_common.models.net_candidate import NetCandidate, NetCandidateWindow, CandidateStatus
from tsn_common.models.profile import CallsignProfile
from tsn_common.models.club import ClubProfile, ClubMembership, ClubRole
from tsn_common.models.trend import TrendSnapshot
from tsn_common.models.support import (
    AiRunLog,
    AnalysisAudit,
    GpuUtilizationSample,
    PhoneticCorrection,
    ProcessingMetric,
    SystemHealth,
)

__all__ = [
    "Base",
    "GUID",
    "AudioFile",
    "AudioFileState",
    "Transcription",
    "TranscriptionBackend",
    "Callsign",
    "CallsignLog",
    "CallsignTopic",
    "ValidationMethod",
    "NetSession",
    "NetParticipation",
    "NetControlSession",
    "CheckinType",
    "NetCandidate",
    "NetCandidateWindow",
    "CandidateStatus",
    "CallsignProfile",
    "ClubProfile",
    "ClubMembership",
    "ClubRole",
    "TrendSnapshot",
    "AiRunLog",
    "AnalysisAudit",
    "GpuUtilizationSample",
    "PhoneticCorrection",
    "ProcessingMetric",
    "SystemHealth",
]
