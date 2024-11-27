class BaseONNXModel(BaseModel):
    def __init__(
        self, model_path: str | Path | list[str], device: Literal["cpu", "cuda"] = "cpu"
    ) -> None:
        super().__init__(model_path=model_path, device=device)

    def load_model(
        self, model_path: str | Path | list[str], device: Literal["cpu", "cuda"] = "cpu"
    ) -> onnxruntime.InferenceSession:
        """
        Load the ONNX model(s) from the given path.

        Parameters
        ----------
        model_path : str | Path | list[str]
            Path to the ONNX model file.

        Returns
        -------
        onnxruntime.InferenceSession
            Inference session(s) object for the loaded model.
        """
        available_providers = onnxruntime.get_available_providers()

        providers = []
        match device:
            case "cuda":
                if ONNXProvider.CUDA.value in available_providers:
                    providers = [ONNXProvider.CUDA.value]
            case "cpu":
                providers = [ONNXProvider.CPU.value]
            case _:
                raise ValueError("No such device!")

        if not providers:
            print("Could not use requested ONNX provider! Using CPU.")
            providers = [ONNXProvider.CPU.value]

        if isinstance(model_path, Path):
            if model_path.is_dir():
                return [
                    onnxruntime.InferenceSession(single_model_path, providers=providers)
                    for single_model_path in model_path.glob("*")
                ]
            return onnxruntime.InferenceSession(model_path, providers=providers)
        elif isinstance(model_path, Iterable):
            return [
                onnxruntime.InferenceSession(single_model_path, providers=providers)
                for single_model_path in model_path
            ]
        else:
            raise ValueError("Wrong type of `model_path`!")
        

class HeadPoseDetector(BaseONNXModel):

    name: str = "fsanet"
    standard: Standard = Standard.REAL_WORLD
    task: Task = Task.HEAD_POSE

    def __init__(
        self, model_path: str | list[str], device: Literal["cpu"] | Literal["cuda"] = "cpu"
    ) -> None:
        super().__init__(model_path=model_path, device=device)
        self.preprocessor = ModelInputPreprocessor(config=HeadPosePreprocessorConfig())

    def detect(self, face_image: np.ndarray) -> HeadPose | None:
        """
        Detects the head pose (yaw, pitch, roll) from the input face image.

        Parameters
        ----------
        face_image : np.ndarray
            Input RGB face image in the form of a NumPy array.

        Returns
        -------
        HeadPose | None
            Detected head pose as a HeadPose object, or None if no head pose is detected.
        """
        image = self.preprocessor(image=face_image)
        yaw_pitch_roll_results = [
            model.run(["output"], {"input": image})[0] for model in self.model
        ]
        yaw, pitch, roll = np.mean(np.vstack(yaw_pitch_roll_results), axis=0)
        return HeadPose(yaw=yaw, pitch=pitch, roll=roll)