INPUT_SCHEMA = {
    "data": {
        'datatype': 'STRING',
        'required': True,
        'shape': [1,-1],
        'example': [["This is a book about stock market.","He wrote an article about renewable energy."]]
    },
    "type": {
        'datatype': 'STRING',
        'required': True,
        'shape': [1],
        'example': ["query"]
    }
}
