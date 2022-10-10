import collections

def unorder_dict(ordered_dict):
   if isinstance(ordered_dict, collections.OrderedDict):
       ordered_dict = dict(ordered_dict)
   for key in ordered_dict:
       if isinstance(ordered_dict[key], collections.OrderedDict):
           ordered_dict[key] = unorder_dict(ordered_dict[key])
   return ordered_dict

def order_dict(d):
   if isinstance(d, dict):
       d = collections.OrderedDict(sorted(d.items()))
   for key in d:
       if isinstance(d[key], dict):
           d[key] = order_dict(d[key])
   return d


def seri_dict(order_dict):
   element_list = []
   def iterate_dict(iter_dict):
       for key in iter_dict:
           if isinstance(iter_dict[key], dict):
               iterate_dict(iter_dict[key])
           else:
               element_list.append(iter_dict[key])
   iterate_dict(order_dict)
   return tuple(element_list)
