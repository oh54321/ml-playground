import abc
from dataclasses import dataclass
from typing import ClassVar, Generic, Optional, Type, TypeVar, get_args

ConfigT = TypeVar("ConfigT", bound="TrainerConfig")


@dataclass(frozen=True)
class TrainerConfig(abc.ABC):
    pass


class TrainingIteration(abc.ABC):
    @abc.abstractmethod
    def step(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def is_done(self) -> bool:
        raise NotImplementedError

    def run(self) -> None:
        while not self.is_done():
            self.step()


class Trainer(abc.ABC, Generic[ConfigT]):
    config_class: ClassVar[Type[TrainerConfig]]

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        config_class = cls._declared_config_class()
        if config_class is not None:
            cls.config_class = config_class
        if cls._has_abstract_methods():
            return
        if getattr(cls, "config_class", None) is None:
            raise TypeError(
                f"{cls.__name__} must specialize its base with a config class, "
                f"e.g. class {cls.__name__}(Trainer[MyConfig])"
            )

    @classmethod
    def _declared_config_class(cls) -> Optional[Type[TrainerConfig]]:
        for base in getattr(cls, "__orig_bases__", ()):
            for arg in get_args(base):
                if isinstance(arg, type) and issubclass(arg, TrainerConfig):
                    return arg
        return None

    @classmethod
    def _has_abstract_methods(cls) -> bool:
        return any(
            getattr(getattr(cls, name, None), "__isabstractmethod__", False)
            for name in dir(cls)
        )

    @abc.abstractmethod
    def create_iteration(self, config: ConfigT) -> TrainingIteration:
        raise NotImplementedError

    def optimize(self, config: ConfigT) -> TrainingIteration:
        iteration = self.create_iteration(config)
        iteration.run()
        return iteration
