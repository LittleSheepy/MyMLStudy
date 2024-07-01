from aidi_vision.aidi_vision import Trainer, Client, Image, LabelIO, Entry, TaskEditor, StringFloatMap


def make_task():
    task_editor = TaskEditor()

    task_editor.set_root_path("D:/denghui_bp/Data/56/Detect_0")  # Set the project root path

    task_editor.set_image_format("bmp")  # Set the project image format

    task_editor.set_indexes([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])  # Set the index (image name) of the training set

    task_editor.set_model_version("V1")  # Set the model version

    task_editor.set_module_name("Detection")  # Set module name

    return task_editor.to_json()


def train():
    trainer = Trainer("daf52973-ce25-11e9-ad13-525400162223")  # create a trainer

    trainer.load_task(make_task())  # load train task from task string

    trainer.start()  # start train, if you want to save model at middle, just start train with args step_by_step = True

    # trainer.step(n) # if step_by_step, must call this function to run n step

    max_iter = trainer.max_iters()

    loss = StringFloatMap()

    for i in range(0, max_iter):

        trainer.get_loss(loss)  # wait get loss of each iter. return an empty map if call times greater than max_iter

        if loss.empty():
            break

        msg = f"iter: {i}"

        for k in loss.keys():
            msg += f", {k}: {loss[k]}"

        print(msg)

    trainer.stop()

    trainer.save_model()  # save trained model


def test():
    client = Client("daf52973-ce25-11e9-ad13-525400162223")

    client.load_task(make_task())

    client.start()

    max_iter = client.max_iters()

    for i in range(0, max_iter):
        id = client.get_current_index()

        print(f"infer on image id: {id}")

    client.stop()


def infer():
    client = Client("daf52973-ce25-11e9-ad13-525400162223")

    # add model with module name and model dir path

    # if your test param file not located at "<prefix>/model/test.json" or "<prefix>/model/<version>/test.json"

    # please specify your test param path like `client.add_model_engine("Segment", "<model_dir>", <param file path>)

    client.add_model_engine("FastDetection", r"C:\Users\KADO\Desktop\13_Side_Black_EWMPS_FS\FastDetection_0\model\V1")
    client.add_model_engine("Segment", r"C:\Users\KADO\Desktop\13_Side_Black_EWMPS_FS\Segment_1\model\V1")

    image = Image()

    image.from_file(r"F:\EWMPS\0.bmp")

    # there many magics to explain

    img_id = client.add_images(image)

    labels, id = client.wait_get_result(img_id)

    image.draw(labels[0])

    image.show(0, "window_name")


def fast_init_serials_model():
    client = Client("daf52973-ce25-11e9-ad13-525400162223")

    # add model with module name and model dir path

    # if your test param file not located at "<prefix>/model/test.json" or "<prefix>/model/<version>/test.json"

    # please specify your test param path like `client.add_model_engine("Segment", "<model_dir>", <param file path>)

    client.add_model_engine("Factory", "<The folder path you set when exporting>")


# Or initialize them one by one in order

# client = Client("daf52973-ce25-11e9-ad13-525400162223")

# client.add_model_engine("FastDetection", "<The folder path you set when exporting>/SUBMODEL_0_FastDetection", "<Optional custom parameter path 1>")

# client.add_model_engine("Segment", "<The folder path you set when exporting>/SUBMODEL_1_Segment", "<Optional custom parameter path 2>")

# client.add_model_engine("Classify", "<The folder path you set when exporting>/SUBMODEL_2_Classify", "<Optional custom parameter path 3>")


if __name__ == "__main__":
    Entry.InitAlgoPlugin()  # Load the algorithm plugin from the given path, if the path parameter is empty, the plugin will be automatically found from the default path<del>This function must be called once before any module-related operation.</del>

    Entry.ModuleNames()  # Return all module names

    Entry.SetLocale("zh_CN")  # Set language (optional zh_TW, zh_CN, en_US)

    Entry.SetLogFilter("info")  # set log filter, should be one of "trace", "debug", "info", "warning", "error", "fatal"

    # Entry.InitLogFile("log_file_path")  # Set the log file path

    Entry.InitLogDir("log_dir_path")  # Set the log directory path

    Entry.GetDeviceNumber()  # Counting gpu numbers

    # train() # Training

    # test() # Testing

    infer()  # Inference

