import logging 

def setup_logger(name="Medical Assistance"):
    logger=logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    ch=logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    format=logging.Formatter("[%(asctime)s] [%(levelname)s]----[%(message)s]")
    ch.setFormatter(format)


    if not logger.hasHandlers():
        logger.addHandler(ch)


    return logger



logger=setup_logger()

logger.info("RAG process Started")



