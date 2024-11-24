import enum


class ErrorMessages(enum.Enum):
    INVALID_REQUEST = "Invalid request"
    INVALID_CREDENTIALS = "Invalid credentials"
    INVALID_DOC_FORMAT = "Invalid document format, only jpg, jpeg, png are allowed"
    SELFIE_STEP_INCOMPLETE = "Please complete selfie step before uplodaing POI"
