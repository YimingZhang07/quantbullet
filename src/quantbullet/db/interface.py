from abc import ABC, abstractmethod
from contextlib import contextmanager
from sqlalchemy.orm import sessionmaker, Session

class TableInterfaceBase( ABC ):
    def __init__(self, engine):
        self.engine = engine
        self.Session = sessionmaker(bind=engine)

    @classmethod
    @abstractmethod
    def DisplayName(self):
        """Return the name of the table."""
        pass

    @property
    @abstractmethod
    def model(self):
        """Return the SQLAlchemy model class."""
        pass

    @contextmanager
    def session_scope(self):
        """Provide a transactional scope around a series of operations."""
        session: Session = self.Session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def get_table_size(self):
        """Get the size of the table."""
        with self.session_scope() as session:
            result = session.query(self.model).count()
        return result
    
    def delete_all_rows(self):
        """Delete all rows from the table."""
        with self.session_scope() as session:
            session.query(self.model).delete()

    def drop_table(self):
        """Drop the table."""
        with self.session_scope() as session:
            self.model.__table__.drop(session.bind)