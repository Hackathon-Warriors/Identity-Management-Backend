import os


class LivelinessCheckerService:
    def __init__(self):
        pass

    @classmethod
    def check_image_liveliness(cls, selfie_image):
        resp = dict(success=False, msg="", error_msg="")
        upload_folder = os.getenv('UPLOAD_FOLDER')
        if not os.path.exists(upload_folder):
            os.mkdir(upload_folder)

        save_path = os.path.join(upload_folder, selfie_image.filename)
        print(save_path)
        selfie_image.save(save_path)
        return resp




