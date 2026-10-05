import pkg.sub.util as u
from .sub.util import async_helper

TIMEOUT = 5


class Base:
    def ping(self):
        return 1

    @property
    def value(self):
        return self._v

    @value.setter
    def value(self, v):
        self._v = v


class Client(Base):
    def __init__(self):
        self.n = u.helper(1)

    async def send(self, request):
        await async_helper(request)
        return self.ping()

    def run(self):
        def inner():
            return self.ping()
        return inner() + super().ping()


def make() -> Client:
    c = Client()
    c.run()
    return c
