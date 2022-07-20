from visual_kinematics.Frame import Frame


class Tool(Frame):
    # ================== Definition and Kinematics
    # a tool is defined by its translation matrix
    # object is derived from Frame and hence support all its operations
    # ==================
    def __init__(self, t_4_4, name: str = ''):
        super().__init__(t_4_4)
        self.name: str = name

    def set_tool_name(self, name: str = ''):
        self.name = name
