from transformers import Tool

class RobotTool(Tool):
    def from_function(function, name, output_type, inputs):
        t = RobotTool()
        t.function = function
        t.name = name
        t.description = function.__doc__
        t.inputs = inputs
        t.output_type = output_type
        return t

    def __call__(self, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], dict):
            return self.function(**args[0])
        return self.function(*args, **kwargs)