class GoodTuringDiscount():
    def __init__(self, coc, gtmax=7, gtmin=1):
        self.gtmax = gtmax
        self.gtmin = gtmin
        self.discount = {}
        A = (self.gtmax + 1) * coc[self.gtmax+1] / coc[1]
        for count in range(self.gtmin, self.gtmax+1):
            cstar = (count + 1) * coc[count+1] / coc[count]
            discount_factor = (cstar / count - A) / (1 - A)
            if discount_factor < 1e-12 or discount_factor > 1:
                discount_factor = 1
            self.discount[count] = count * discount_factor
            # discounted value can then be used to compute f(a_z)
            # f(a_z) = discount[count[a_z]] / count[a_]

    def __call__(self, count):
        if count > self.gtmax:
            # considered reliable
            return count
        elif count < self.gtmin:
            return 0
        else:
            return self.discount[count]
