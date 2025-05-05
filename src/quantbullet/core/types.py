from enum import Enum

class DataType(Enum):
    DATE            = "date"
    STRING          = "string"
    FLOAT           = "float"
    INT             = "int"
    BOOL            = "bool"
    DATETIME        = "datetime"
    CATEGORICAL     = "categorical"
    
    @classmethod
    def numeric_types(cls):
        return {cls.FLOAT, cls.INT}
    
    def is_numeric(self):
        return self in self.numeric_types()
    
    def is_categorical(self):
        return self == self.CATEGORICAL