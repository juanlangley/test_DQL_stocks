

import numpy as np

class OrderBuffer(object):
    def __init__(self):
        self.order_buffer = []
    
    def store(self, Open_Time, Close_Time, Order_Type, Pips, Profit):
        self.order_buffer.append([Open_Time, Close_Time, Order_Type, Pips, Profit])

    def get_size(self):
        return len(self.order_buffer)
    
    def get_buffer(self):
        return np.array(self.order_buffer)
    
    def reset_buffer(self):
        self.order_buffer.clear()
    
    def export_buffer(self):
        # Numerical 2D array
        array = np.array(self.order_buffer)
        
        rows = ["{}/{}/{}/{}/{}".format(i, j, k, l, m) for i, j, k, l, m in array] 
        text = "\n".join(rows) 
          
        with open('orders/orders_file.csv', 'w') as f: 
            f.write(text)



"""
buffer = orderBuffer()

buffer.store('2021-05-03 01:57:00', '2021-05-03 01:57:00', "indefinido", 0, 0)

buffer.store('2023-01-01 00:00:00', '2023-05-01 00:00:00',  "sell",  20, 2)
buffer.store('1999-01-01 00:00:00', '2001-05-01 00:00:00',  "buy",  55, 5.5)


listBuffer = buffer.order_buffer

array = np.array(buffer.order_buffer)

buffer.export_buffer()

print(buffer.get_size())

start_date = '2023-01-01  00:00:00'
end_date = '2023-05-01 00:00:00'
"""