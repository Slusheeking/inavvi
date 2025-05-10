"""Type stubs for SQLAlchemy."""

from typing import Any, Callable, Dict, List, Optional, TypeVar, Union, overload

# Core SQLAlchemy types
BigInteger: Any
Column: Any
DateTime: Any
Float: Any
Index: Any
Integer: Any
MetaData: Any
PrimaryKeyConstraint: Any
String: Any
create_engine: Any
select: Any
text: Any
func: Any

# Declare ORM classes
def declarative_base(*args: Any, **kwargs: Any) -> Any: ...

# Declare modules
class ext:
    class declarative:
        def declarative_base(*args: Any, **kwargs: Any) -> Any: ...

class orm:
    Session: Any
    sessionmaker: Any

class pool:
    QueuePool: Any

class dialects:
    class postgresql:
        insert: Any
