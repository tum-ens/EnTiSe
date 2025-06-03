import math

def get_with_backup(obj, key, backup = None):
    value = obj.get(key, backup)
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return backup
    return value