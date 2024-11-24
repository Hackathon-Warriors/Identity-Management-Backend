import enum


class LivelinessRequestStatus(enum.Enum):
    INITIATED = "initiated"
    SUCCESS = "success"
    FAILED = "failed"


class DocumentTypes(enum.Enum):
    POI = 'POI'
    BANK_STATEMENT = 'BANK_STATEMENT'
    ITR = 'ITR'
