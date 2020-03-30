

def hypercube(self, k):
    return np.all(k < 0.5, axis=1)


def radial(self, k):
    const_part = (2 * np.pi) ** (-self.dim / 2)
    return const_part * np.exp(-0.5 * np.add.reduce(k ** 2, axis=1))