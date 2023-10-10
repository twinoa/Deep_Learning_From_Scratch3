import numpy as np
import weakref
import contextlib


class Config:
    enable_backprop = True


class Variable:
    def __init__(self, data, name=None):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{}은(는) 지원하지 않습니다.'.format(type(data)))

        self.data = data
        self.name = name
        self.grad = None
        self.creator = None
        self.generation = 0  # 세대 수를 기록하는 변수
    
    __array_priority__ = 200

    @property
    def shape(self):
        return self.data.shape
    
    @property
    def ndim(self):
        return self.data.size
    
    @property
    def dtye(self):
        return self.data.dtype
    
    def __len__(self):
        return len(self.data)
    
    def __repr__(self):
        if self.data is None:
            return "variable(None)"
        p = str(self.data).replace('\n', '\n' + ' ' * 9)
        return 'variable(' + p + ')'
    
    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1  # 세대를 기록한다 (부모세대 + 1)

    def backward(self, retain_grad=False, create_graph=False):
        if self.grad is None:
            # self.grad = np.ones_like(self.data)
            self.grad = Variable(np.ones_like(self.data))

        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)

        while funcs:
            f = funcs.pop()  # 1. 함수를 가져온다.
            gys = [output().grad for output in f.outputs]

            with using_config('enable_backprop', create_graph):
                gxs = f.backward(*gys)  # 메인 backward
                if not isinstance(gxs, tuple):
                    gxs = (gxs,)

                for x, gx in zip(f.inputs, gxs):
                    if x.grad is None:
                        x.grad = gx
                    else :
                        x.grad = x.grad + gx

                    if x.creator is not None:
                        add_func(x.creator)

            if not retain_grad:
                for y in f.outputs:
                    y().grad = None  # y는 약한 참조 

    def cleargrad(self):
        self.grad = None


class Function:
    def __call__(self, *inputs):
        inputs = [as_variable(x) for x in inputs]

        xs = [x.data for x in inputs]
        ys = self.forward(*xs)  # 별표를 붙여 언팩
        if not isinstance(ys, tuple):  # 튜플이 아닌 경우 추가 지원
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])  # 세대 설정
            for output in outputs:
                output.set_creator(self)  # 연결 설정
            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]

        # 리스트의 원소가 하나라면 첫 번째 원소를 반환한다.
        return outputs if len(outputs) > 1 else outputs[0]
    
        
    def forward(self, x):
        raise NotImplementedError()
    
    def backward(self, gy):
        raise NotImplementedError
    

class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return (y,)
    
    def backward(self, gy):
        return gy, gy


class Mul(Function):
    def forward(self, x0, x1):
        y = x0 * x1
        return y
    
    def backward(self, gy):
        x0, x1 = self.inputs
        return gy * x1, gy * x0
    

class Neg(Function):
    def forward(self, x):
        return -x
    
    def backward(self, gy):
        return -gy


class Sub(Function):
    def forward(self, x0, x1):
        y = x0 - x1
        return y
    
    def backward(self, gy):
        return gy, -gy
    

class Div(Function):
    def forward(self, x0, x1):
        y = x0 / x1
        return y
    
    def backward(self, gy):
        x0, x1 = self.inputs
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1 ** 2)
        return gx0, gx1
    

class Pow(Function):
    def __init__(self, c):
        self.c = c

    def forward(self, x):
        y = x ** self.c
        return y
    
    def backward(self, gy):
        x = self.inputs[0]
        c = self.c
        gx = c * x ** (c - 1) * gy
        return gx
    

@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)


def no_grad():
    return using_config('enable_backprop', False)


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)


def add(x0, x1):
    x1 = as_array(x1)
    return Add()(x0, x1)


def mul(x0, x1):
    x1 = as_array(x1)
    return Mul()(x0, x1)


def neg(x):
    return Neg()(x)


def sub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x0, x1)


def rsub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x1, x0)  # x0와 x1의 순서를 바꾼다.


def div(x0, x1):
    x1 = as_array(x1)
    return Div()(x0, x1)


def rdiv(x0, x1):
    x1 = as_array(x1)
    return Div()(x1, x0)  # x0와 x1의 순서를 바꾼다.


def pow(x, c):
    return Pow(c)(x)


def setup_variable():
    Variable.__rmul__ = mul
    Variable.__mul__ = mul
    Variable.__radd__ = add
    Variable.__add__ = add
    Variable.__neg__ = neg
    Variable.__sub__ = sub
    Variable.__rsub__ = rsub
    Variable.__truediv__ = div
    Variable.__rtruediv__ = rdiv
    Variable.__pow__ = pow