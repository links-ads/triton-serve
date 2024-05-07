from logging.config import fileConfig
from time import sleep

from alembic import context
from sqlalchemy import engine_from_config, pool

from triton_serve.config import get_settings
from triton_serve.database.model import Base  # noqa

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
settings = get_settings()
config = context.config
config.set_main_option("sqlalchemy.url", settings.database_url)
target_metadata = Base.metadata


# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    for i in range(3):
        try:
            connectable = engine_from_config(
                config.get_section(config.config_ini_section, {}),
                prefix="sqlalchemy.",
                poolclass=pool.NullPool,
            )
            with connectable.connect() as connection:
                context.configure(connection=connection, target_metadata=target_metadata)

                with context.begin_transaction():
                    context.run_migrations()
            break
        except Exception as e:
            # sleep an increasing amount of time
            print(f"Failed to connect to database: {e}")
            print(f"Retrying connection... Attempt {i + 1} of 3")
            seconds = 2 ** (i + 1)
            sleep(seconds)


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
