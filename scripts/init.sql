CREATE TYPE work_status AS ENUM ('Pending', 'In progress', 'Finished', 'Failed');
CREATE TYPE segment_stage as ENUM ('Translation', 'Voice over');
CREATE TYPE job_stage as ENUM ('Processing segments', 'Merge');


CREATE TABLE jobs (
    id INT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    status work_status NOT NULL DEFAULT 'Pending',

    submited_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    finished_at TIMESTAMP WITH TIME ZONE DEFAULT NULL

);

CREATE TABLE job_stages (
    stage job_stage NOT NULL,
    job_id INT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,

    started_at TIMESTAMP WITH TIME ZONE DEFAULT NULL,
    finished_at TIMESTAMP WITH TIME ZONE DEFAULT NULL,

    status work_status NOT NULL DEFAULT 'Pending'
);


CREATE TABLE segments (
    id INT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    job_id INT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,

    start_time NUMERIC(10, 3) NOT NULL,
    end_time NUMERIC(10, 3) NOT NULL
);


CREATE TABLE segment_stages (
    stage segment_stage NOT NULL,
    segment_id INT NOT NULL REFERENCES segments(id) ON DELETE CASCADE,

    started_at TIMESTAMP WITH TIME ZONE DEFAULT NULL,
    finished_at TIMESTAMP WITH TIME ZONE DEFAULT NULL,

    status work_status NOT NULL DEFAULT 'Pending'
);
