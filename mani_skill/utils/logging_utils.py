import logging

# https://github.com/openai/gym/blob/master/gym/utils/colorize.py
color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38,
)


def colorize(
    string: str, color: str, bold: bool = False, highlight: bool = False
) -> str:
    """Returns string surrounded by appropriate terminal colour codes to print colourised text.
    Args:
        string: The message to colourise
        color: Literal values are gray, red, green, yellow, blue, magenta, cyan, white, crimson
        bold: If to bold the string
        highlight: If to highlight the string
    Returns:
        Colourised string
    """
    attr = []
    num = color2num[color]
    if highlight:
        num += 10
    attr.append(str(num))
    if bold:
        attr.append("1")
    attrs = ";".join(attr)
    return f"\x1b[{attrs}m{string}\x1b[0m"


class CustomFormatter(logging.Formatter):
    # https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output

    def format(self, record):
        # Store the original levelname
        levelname_orig = record.levelname
        
        # Color only the levelname based on severity
        if record.levelno == logging.WARNING:
            record.levelname = colorize(levelname_orig, "yellow", True)
        elif record.levelno == logging.ERROR:
            record.levelname = colorize(levelname_orig, "red", True)
        elif record.levelno == logging.INFO:
            record.levelname = colorize(levelname_orig, "green")
        elif record.levelno == logging.DEBUG:
            record.levelname = colorize(levelname_orig, "blue")
        
        # Format the entire message (only levelname is colored)
        s = super().format(record)
        
        # Restore original levelname
        record.levelname = levelname_orig
        return s


logger = logging.getLogger("mani_skill ")
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.hasHandlers():
    ch = logging.StreamHandler()
    ch.setFormatter(
        CustomFormatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(ch)
