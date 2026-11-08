import enum
import os
import urllib.parse
from contextlib import contextmanager
from datetime import UTC, datetime, timezone
from decimal import Decimal

from sqlalchemy import (
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    Numeric,
    String,
    create_engine,
    delete,
    func,
    update,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker

DB_IP = os.getenv("IP")
DB_PORT = os.getenv("DB_PORT")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = urllib.parse.quote_plus(os.getenv("DB_PASSWORD"))
DB_NAME = os.getenv("DB_NAME")

# "postgresql://user:password@localhost:5432/dbname"
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_IP}:{DB_PORT}/{DB_NAME}"

engine = None
SessionLocal = None

class Base(DeclarativeBase):
    pass

class WorkStatus(enum.StrEnum):
    PENDING = "Pending"
    IN_PROGRESS = "In progress"
    FINISHED = "Finished"
    FAILED = "Failed"

class SegmentStageType(enum.StrEnum):
    TRANSLATION = "Translation"
    VOICE_OVER = "Voice over"

class JobStageType(enum.StrEnum):
    MERGE = "Merge"
    PROCESSING_SEGMENTS = "Processing segments"


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    status: Mapped[WorkStatus] = mapped_column(
        Enum(
            WorkStatus,
            name="work_status",
            create_type=False,
            values_callable=lambda x: [e.value for e in x],
        ),
        default=WorkStatus.PENDING,
        nullable=False,
    )

    submited_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.now(UTC),
        nullable=False
    )

    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime,
        default=None,
        nullable=True
    )

    segments: Mapped[list["Segment"]] = relationship(
        back_populates="job",
        cascade="all, delete-orphan"
    )

    job_stages: Mapped[list["JobStage"]] = relationship(
        back_populates="job",
        cascade="all, delete-orphan"
    )

    def to_dict(self) -> dict:
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}

class JobStage(Base):
    __tablename__ = "job_stages"

    stage: Mapped[JobStageType] = mapped_column(
        Enum(
            JobStageType,
            name="stage",
            create_type=False,
            values_callable=lambda x: [e.value for e in x],
        ),
        primary_key=True,
    )

    job_id: Mapped[int] = mapped_column(
        ForeignKey(
            "jobs.id",
            ondelete="CASCADE"
        ),
        primary_key=True
    )

    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, default=None
    )

    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, default=None
    )

    status: Mapped[WorkStatus] = mapped_column(
        Enum(
            WorkStatus,
            name="work_status",
            create_type=False,
            values_callable=lambda x: [e.value for e in x],
        ),
        nullable=False,
        default=WorkStatus.PENDING,
    )

    job: Mapped["Job"] = relationship(
        back_populates="job_stages"
    )

    def to_dict(self) -> dict:
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}


class Segment(Base):
    __tablename__ = "segments"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True
    )

    job_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey(
            "jobs.id",
            ondelete="CASCADE"
        ),
        nullable=False
    )

    start_time: Mapped[Decimal] = mapped_column(
        Numeric(10, 3),
        nullable=False
    )

    end_time: Mapped[Decimal] = mapped_column(
        Numeric(10, 3),
        nullable=False
    )

    job: Mapped["Job"] = relationship(
        back_populates="segments"
    )

    segment_stages: Mapped[list["SegmentStage"]] = relationship(
        back_populates="segment",
        cascade="all, delete-orphan"
    )

    def to_dict(self) -> dict:
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}


class SegmentStage(Base):
    __tablename__ = "segment_stages"

    stage: Mapped[SegmentStageType] = mapped_column(
        Enum(
            SegmentStageType,
            name="stage",
            create_type=False,
            values_callable=lambda x: [e.value for e in x],
        ),
        primary_key=True,
    )

    segment_id: Mapped[int] = mapped_column(
        ForeignKey(
            "segments.id",
            ondelete="CASCADE"
        ),
        primary_key=True
    )

    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, default=None
    )

    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, default=None
    )

    status: Mapped[WorkStatus] = mapped_column(
        Enum(
            WorkStatus,
            name="work_status",
            create_type=False,
            values_callable=lambda x: [e.value for e in x],
        ),
        nullable=False,
        default=WorkStatus.PENDING,
    )

    segment: Mapped["Segment"] = relationship(
        back_populates="segment_stages"
    )

    def to_dict(self) -> dict:
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}


def init_db():
    """
    Performs db initialisation. Inti done once on worker init
    """
    global engine, SessionLocal
    if engine is None:
        engine = create_engine(DATABASE_URL, pool_size=5, max_overflow=10, pool_pre_ping=True)
        SessionLocal = sessionmaker(bind=engine)


@contextmanager
def get_db_session():
    """
    Context manager for db transactions
    """
    if SessionLocal is None:
        raise RuntimeError("db not initialised, call init_db first.")

    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_job_by_id(session, job_id: int) -> Job | None:
    return session.get(Job, job_id)

def get_job_stage_by_id(session, stage_type: JobStageType, job_id: int) -> JobStage | None:
    return session.get(JobStage, (stage_type, job_id))

def get_segment_by_id(session, segment_id: int) -> Segment | None:
    return session.get(Segment, segment_id)

def get_segment_stage_by_id(session, stage_type: SegmentStageType,
                            segment_id: int) -> SegmentStage | None:
    return session.get(SegmentStage, (stage_type, segment_id))


def insert_job(session, status: WorkStatus = WorkStatus.PENDING) -> int:

    new_job = Job(status=status)

    session.add(new_job)

    session.flush()

    return new_job.id

def insert_job_stage(session, stage_type: JobStageType, job_id: int,
                     status: WorkStatus = WorkStatus.PENDING) -> tuple[JobStageType, int]:
    new_stage = JobStage(stage=stage_type, job_id=job_id, status=status)
    session.add(new_stage)
    session.flush()

    return (new_stage.stage, new_stage.job_id)


def insert_segment(session, job_id: int, start_time: float, end_time: float) -> int:
    new_segment = Segment(
        job_id=job_id,
        start_time=Decimal(str(start_time)),
        end_time=Decimal(str(end_time))
    )

    session.add(new_segment)
    session.flush()
    return new_segment.id


def insert_segment_stage(session, stage_type: SegmentStageType, segment_id: int,
                         status: WorkStatus = WorkStatus.PENDING) -> tuple[SegmentStageType, int]:
    new_stage = SegmentStage(stage=stage_type, segment_id=segment_id, status=status)
    session.add(new_stage)
    session.flush()

    return (new_stage.stage, new_stage.segment_id)

def clear_all_tables(session) -> None:
    session.execute(delete(Job))
    session.flush()


def update_job_status(session, job_id: int, new_status: WorkStatus) -> None:
    stmt = (
        update(Job)
        .where(Job.id == job_id)
        .values(status=new_status)
    )
    session.execute(stmt)
    session.flush()

def update_job_stage_status(session, stage_type: JobStageType,
                            job_id: int, new_status: WorkStatus) -> None:
    stmt = (
        update(JobStage)
        .where(JobStage.stage == stage_type, JobStage.job_id == job_id)
        .values(status=new_status)
    )
    session.execute(stmt)
    session.flush()

def update_segment_stage_status(session, stage_type: SegmentStageType,
                                segment_id: int, new_status: WorkStatus) -> None:
    stmt = (
        update(SegmentStage)
        .where(SegmentStage.stage == stage_type, SegmentStage.segment_id == segment_id)
        .values(status=new_status)
    )
    session.execute(stmt)
    session.flush()


def set_job_finished_time(session, job_id: int,
                          finished_at: datetime | None = None) -> None:
    stmt = (
        update(Job)
        .where(Job.id == job_id)
        .values(finished_at=finished_at if finished_at else func.now())
    )
    session.execute(stmt)
    session.flush()

def set_job_stage_started_time(session, stage_type: JobStageType,
                               job_id: int, started_at: datetime | None = None) -> None:
    stmt = (
        update(JobStage)
        .where(JobStage.stage == stage_type, JobStage.job_id == job_id)
        .values(started_at=started_at if started_at else func.now())
    )
    session.execute(stmt)
    session.flush()

def set_segment_stage_started_time(session, stage_type: SegmentStageType,
                                   segment_id: int, started_at: datetime | None = None) -> None:
    stmt = (
        update(SegmentStage)
        .where(SegmentStage.stage == stage_type, SegmentStage.segment_id == segment_id)
        .values(started_at=started_at if started_at else func.now())
    )
    session.execute(stmt)
    session.flush()

def set_stage_finished_time(session, stage_type: SegmentStageType,
                            segment_id: int, finished_at: datetime | None = None) -> None:
    stmt = (
        update(SegmentStage)
        .where(SegmentStage.stage == stage_type, SegmentStage.segment_id == segment_id)
        .values(finished_at=finished_at if finished_at else func.now())
    )
    session.execute(stmt)
    session.flush()
