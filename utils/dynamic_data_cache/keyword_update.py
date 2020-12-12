from utils.dynamic_data_cache.keyword_dao import keywordinital
from utils.logger import logger


def keyword_operate(operate_code, keyword_tuple_list):
    if operate_code == -1:  # 删除
        logger.info(keyword_tuple_list)
        keywordinital.delete(keyword_tuple_list)


    elif operate_code == 1:  # 新增
        logger.info(keyword_tuple_list)
        keywordinital.add(keyword_tuple_list)

    elif operate_code == 2:  # 更新所有关键词
        logger.info(keyword_tuple_list)
        keywordinital.update_all()

    else:
        raise NotImplementedError('{} is not a vaild '
                            'operate code.'.format(operate_code))

