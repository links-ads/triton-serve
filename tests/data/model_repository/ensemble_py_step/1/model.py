import triton_python_backend_utils as pb_utils


class TritonPythonModel:

    @staticmethod
    def auto_complete_config(auto_complete_model_config):
        return auto_complete_model_config

    def initialize(self, args):
        print("Initialized...")

    def execute(self, requests):
        responses = []
        for request in requests:
            batch = pb_utils.get_input_tensor_by_name(request, "input_img").as_numpy()
            out_tensor = pb_utils.Tensor("preprocessed_input", batch)
            response = pb_utils.InferenceResponse(output_tensors=[out_tensor])
            responses.append(response)
        return responses

    def finalize(self):
        print("Cleaning up...")
