"""离线物理模型身份；这里不推断任何几何或审查。没有模型的历史配置文件仍然是UR10e配置文件。
明确选择新模型需要进行匹配的单独身份审查。此绑定不会使安装或复制的路由在物理上被接受。"""

SUPPORTED_MODELS = ('ur5e', 'ur10e')


def model_for_profile(data, default='ur10e'):
    """返回经过审查的明确模型，或历史遗留的默认值。"""
    if default not in SUPPORTED_MODELS:
        raise ValueError('Unsupported physical robot model default')
    if not isinstance(data, dict):
        raise ValueError('Physical robot model profile required')
    if 'robot_model' not in data:
        return default
    model = data['robot_model']
    if model not in SUPPORTED_MODELS:
        raise ValueError('Unsupported physical robot model')
    if data.get('reviewed_robot_model') != model:
        raise ValueError('Physical robot model must match its explicit review')
    return model


def dashboard_model_names(model):
    """仪表板别名仍然需要单独进行PolyScope 5 e-Series检查。"""
    if model not in SUPPORTED_MODELS:
        raise ValueError('Unsupported physical robot model')
    return ('UR5', 'UR5e') if model == 'ur5e' else ('UR10', 'UR10e')
