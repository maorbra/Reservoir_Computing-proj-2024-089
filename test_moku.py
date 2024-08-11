from moku.instruments import Oscilloscope, FIRFilterBox

ip = '[fe80::7269:79ff:feb9:45f6%3]'

osc = Oscilloscope(ip, force_connect=True)
fir = FIRFilterBox(ip, force_connect=True)
