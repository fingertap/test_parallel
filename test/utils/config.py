import configparser
from abc import ABC
from typing import AnyStr, List, Union, Any


def _type_cast(variable: AnyStr) -> Union[AnyStr, int, float, bool]:
    variable = variable.strip()
    if variable.lower() == 'true':
        return True
    if variable.lower() == 'false':
        return False
    try:
        return int(variable)
    except ValueError:
        try:
            return float(variable)
        except ValueError:
            return variable


class ConfigModule(ABC):
    def __init__(self, config_files: Union[AnyStr, List[AnyStr]] = None):
        self._config = configparser.ConfigParser()
        self._config.read(config_files or [])
        # NOTE: configure文件中的变量并不区分大小写

    def read_config(self,
                    section: AnyStr,
                    field: AnyStr = None,
                    fallback: Any = None
                    ) -> Any:
        if field is None:
            section, field = 'default', section
        if section.lower() != 'default' and fallback is None:
            if 'default' in self._config and field in self._config['default']:
                fallback = _type_cast(self._config.get('default', field))
        if section not in self._config or field not in self._config[section]:
            return fallback
        return _type_cast(self._config.get(section, field))
