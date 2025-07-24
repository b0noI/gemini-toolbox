from google.generativeai import GenerativeModel

class LLM:
    def __init__(self, model_name='gemini-pro'):
        self.client = GenerativeModel(model_name)
    
    def generate(self, prompt, stream=False, **kwargs):
        try:
            response = self.client.generate_content(
                prompt,
                stream=stream,
                **kwargs
            )
            return self._process_response(response, stream=stream)
        except Exception as e:
            return self._error_response(str(e), stream)



    def _process_response(self, response, stream=False):
        if stream:
            for chunk in response:
                yield {
                    "content": chunk.text,
                    "is_complete": False
                }
            yield {"content": "", "is_complete": True}
        else:
            return {"content": response.text, "is_complete": True}
        
    def _error_response(self, message, stream):
        if stream:
            def error_generator():
                yield {"error": message, "is_complete": True}
            return error_generator()
        else:
            return [{"error": message, "is_complete": True}]
