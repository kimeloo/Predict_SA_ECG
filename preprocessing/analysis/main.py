import logging

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
log_formatter = logging.Formatter("[%(asctime)s] [%(levelname)8s] : [%(name)15s] --- %(message)s")
log_handler = logging.StreamHandler()
log_handler.setLevel(logging.CRITICAL)
log_handler.setFormatter(log_formatter)
log_warning_formatter = logging.Formatter("[%(asctime)s] [%(levelname)8s] : [%(name)15s] --- %(message)s")
log_warning_handler = logging.FileHandler("event.log")
log_warning_handler.setLevel(logging.INFO)
log_warning_handler.setFormatter(log_warning_formatter)
logger.addHandler(log_handler)
logger.addHandler(log_warning_handler)

if __name__ == '__main__':
    logger.info('running...')
    # from . import edf_sampling_rate
    # edf_sampling_rate.run()

    from . import edf_quality_check
    edf_quality_check.run()