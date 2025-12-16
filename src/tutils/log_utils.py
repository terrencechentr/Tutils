import logging
import sys
from colorlog import ColoredFormatter

class Colors:
    """ANSI color codes for custom log coloring"""
    RESET = "\033[0m"
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    
    # Bright/Bold colors
    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"
    
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"

class ColoredLogger(logging.Logger):
    def info(self, msg, color="GREEN", *args, **kwargs):
        """
        Log an info message with specific color.
        
        Args:
            msg: The message to log
            color: The color to use. Can be a color name string (e.g. "RED", "BLUE") 
                   found in Colors class, or an ANSI color code.
                   Defaults to "GREEN".
            *args, **kwargs: Arguments passed to logging.Logger.info
        """
        if isinstance(color, str):
            # Try to find the color in Colors class (case-insensitive)
            color_upper = color.upper()
            if hasattr(Colors, color_upper):
                color_code = getattr(Colors, color_upper)
            else:
                # Assume it might be a raw ANSI code if not found
                color_code = color
        else:
            color_code = str(color)
            
        # Reset color at the end
        colored_msg = f"{color_code}{msg}{Colors.RESET}"
        super().info(colored_msg, *args, **kwargs)

# Register our custom logger class
logging.setLoggerClass(ColoredLogger)

# Add %(log_color)s to the format string so color codes take effect
FORMATTER = ColoredFormatter(
    "%(log_color)s[%(asctime)s] [%(levelname)s] [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    log_colors={
        "DEBUG": "cyan",
        "INFO": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "bold_red"
    },
    reset=True,
    style='%'
)

LEVEL_MAP = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL
}

def get_logger(name: str = "root", level: str = "INFO") -> ColoredLogger:
    logger = logging.getLogger(name)
    # Force class update if it's a standard Logger (safety check)
    if not isinstance(logger, ColoredLogger):
        logger.__class__ = ColoredLogger
        
    if level not in LEVEL_MAP:
        raise ValueError(f"Invalid log level: {level}")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(FORMATTER)
        logger.addHandler(handler)
        
    logger.setLevel(LEVEL_MAP[level])
    logger.propagate = False
    
    return logger
