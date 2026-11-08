import os

import media_dub.common.database as db

if __name__ == "__main__":
    db.init_db()

    with db.get_db_session() as session:


        id = db.insert_job(session)

        print(id)

        seg_id = db.insert_segment(session, id, 123, 214)

        print(seg_id)

        stage_res = db.insert_segment_stage(session, "Translation", seg_id)

        print(stage_res)

        job_stage_res = db.insert_job_stage(session, "Processing segments", id)

        print(job_stage_res)



        job = db.get_job_by_id(session, id)

        job = job.to_dict()

        print(job)


        segment = db.get_segment_by_id(session, seg_id)

        segment = segment.to_dict()

        print(segment)

        stage = db.get_segment_stage_by_id(session, stage_res[0], stage_res[1])

        stage = stage.to_dict()

        print(stage)


        stage = db.get_job_stage_by_id(session, job_stage_res[0], job_stage_res[1])

        stage = stage.to_dict()

        print(stage)


        db.clear_all_tables(session)



