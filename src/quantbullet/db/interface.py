from abc import ABC, abstractmethod
from contextlib import contextmanager
from sqlalchemy.orm import sessionmaker, Session

class TableInterfaceBase( ABC ):
    """Base class for SQLAlchemy table interfaces."""
    def __init__(self, engine):
        self.engine = engine
        self.Session = sessionmaker(bind=engine)

    @property
    @abstractmethod
    def model(self):
        """Return the SQLAlchemy model class.
        
        This abstract method must be implemented by subclasses, and return the SQLAlchemy model class (the ORM mapping).
        """
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