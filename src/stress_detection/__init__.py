from loguru import logger
import sys

logger.remove()

if "--debug" in sys.argv:
    logger.add("logs_{time}.log")
# Specify the file path and format for the log file
logger.add("info.log", format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}")