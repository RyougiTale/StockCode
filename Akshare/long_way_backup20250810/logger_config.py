import logging
import logging.config
import os
from datetime import datetime
import sys


class ColoredFormatter(logging.Formatter):
    """带颜色的日志格式化器，在终端中显示彩色日志"""
    
    COLORS = {
        'DEBUG': '\033[36m',     # 青色
        'INFO': '\033[32m',      # 绿色  
        'WARNING': '\033[33m',   # 黄色
        'ERROR': '\033[31m',     # 红色
        'CRITICAL': '\033[35m'   # 紫色
    }
    RESET = '\033[0m'
    
    def format(self, record):
        log_message = super().format(record)
        color = self.COLORS.get(record.levelname, self.RESET)
        return f"{color}{log_message}{self.RESET}"


def setup_logging(
    log_level=logging.INFO,
    log_file=None,
    enable_console=True,
    enable_file=True,
    max_file_size=10*1024*1024,  # 10MB
    backup_count=5
):
    """
    设置项目的日志系统
    
    Args:
        log_level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: 日志文件路径，如果为None则自动生成
        enable_console: 是否启用控制台输出
        enable_file: 是否启用文件输出
        max_file_size: 日志文件最大大小（字节）
        backup_count: 保留的备份文件数量
    """
    
    # 创建日志目录
    log_dir = os.path.join(os.path.dirname(__file__), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # 自动生成日志文件名
    if log_file is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'stock_prediction_{timestamp}.log')
    
    # 日志配置字典
    config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'detailed': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            },
            'simple': {
                'format': '%(levelname)s - %(name)s - %(message)s'
            },
            'colored': {
                '()': ColoredFormatter,
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            }
        },
        'handlers': {},
        'root': {
            'level': log_level,
            'handlers': []
        }
    }
    
    # 添加控制台处理器
    if enable_console:
        config['handlers']['console'] = {
            'class': 'logging.StreamHandler',
            'level': log_level,
            'formatter': 'colored' if sys.stdout.isatty() else 'simple',
            'stream': sys.stdout
        }
        config['root']['handlers'].append('console')
    
    # 添加文件处理器
    if enable_file:
        config['handlers']['file'] = {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': logging.DEBUG,  # 文件记录更详细的日志
            'formatter': 'detailed',
            'filename': log_file,
            'maxBytes': max_file_size,
            'backupCount': backup_count,
            'encoding': 'utf-8'
        }
        config['root']['handlers'].append('file')
    
    # 应用配置
    logging.config.dictConfig(config)
    
    # 设置第三方库的日志级别，避免过多输出
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    
    return logging.getLogger(__name__)


def get_logger(name=None):
    """获取日志记录器"""
    if name is None:
        import inspect
        frame = inspect.currentframe().f_back
        name = frame.f_globals.get('__name__', 'unknown')
    
    return logging.getLogger(name)


# 便利函数，用于快速替换print语句
def log_info(message, *args, **kwargs):
    """替换print的便利函数"""
    logger = get_logger()
    logger.info(message, *args, **kwargs)


def log_debug(message, *args, **kwargs):
    """调试信息"""
    logger = get_logger()
    logger.debug(message, *args, **kwargs)


def log_warning(message, *args, **kwargs):
    """警告信息"""
    logger = get_logger()
    logger.warning(message, *args, **kwargs)


def log_error(message, *args, **kwargs):
    """错误信息"""
    logger = get_logger()
    logger.error(message, *args, **kwargs)


# 性能监控装饰器
def log_performance(func_name=None):
    """性能监控装饰器"""
    def decorator(func):
        import time
        import functools
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            name = func_name or f"{func.__module__}.{func.__name__}"
            logger = get_logger()
            
            start_time = time.time()
            logger.debug(f"开始执行 {name}")
            
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                logger.info(f"{name} 执行完成，耗时: {execution_time:.4f}秒")
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"{name} 执行失败，耗时: {execution_time:.4f}秒，错误: {e}")
                raise
        
        return wrapper
    return decorator


if __name__ == '__main__':
    # 测试日志系统
    setup_logging(log_level=logging.DEBUG)
    logger = get_logger()
    
    logger.debug("这是调试信息")
    logger.info("这是普通信息")
    logger.warning("这是警告信息")
    logger.error("这是错误信息")
    
    # 测试性能监控
    @log_performance("测试函数")
    def test_function():
        import time
        time.sleep(0.1)
        return "测试完成"
    
    result = test_function()
    print(f"函数返回: {result}")