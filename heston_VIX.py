import numpy as np
import matplotlib.pyplot as plt
import QuantLib as ql


class HestonParams:
    def __init__(self, S0=100, V0=0.04, r=0.025, kappa=1.5, theta=0.04, sigma=0.30, rho=-0.9):
        self.S0 = S0
        self.V0 = V0
        self.r = r
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho


def Simulate_Heston_QuantLib(NumOfAssets=10, TimeSteps=30, dt=1. / 365,
                             S0=100, V0=0.04, r=0.025, kappa=1.5, theta=0.04, sigma=0.30,
                             rho=-0.9):  # V0=0.48, kappa=1.2, theta=0.25, sigma=0.80, rho=-0.64
    today = ql.Date(15, ql.October, 2008)
    riskFreeRate = ql.FlatForward(today, r, ql.ActualActual())
    dividendRate = ql.FlatForward(today, 0.0, ql.ActualActual())
    # discretization = Reflection  # PartialTrunction, FullTruncation, Reflection, ExactVariance
    hp = ql.HestonProcess(ql.YieldTermStructureHandle(riskFreeRate),
                          ql.YieldTermStructureHandle(dividendRate),
                          ql.QuoteHandle(ql.SimpleQuote(S0)),  # S0
                          V0,  # v0
                          kappa,  # kappa
                          theta,  # theta
                          sigma,  # sigma
                          rho,  # rho
                          )
    times = [n * dt for n in range(TimeSteps)]
    rsg = ql.GaussianRandomSequenceGenerator(
        ql.UniformRandomSequenceGenerator(2 * (len(times) - 1), ql.UniformRandomGenerator()))
    mpg = ql.GaussianMultiPathGenerator(hp, times, rsg, brownianBridge=False)

    S = np.zeros((len(times), NumOfAssets))
    V = np.zeros((len(times), NumOfAssets))
    for i in range(NumOfAssets):
        sample = mpg.next()
        multipath = sample.value()
        S[:, i] = multipath[0]
        V[:, i] = multipath[1]
    return S, V


def price(S0=100, K=100, V0=0.04, r=0.025, kappa=1, theta=0.04, sigma=0.8, rho=-0.7, TimeSteps=30):

    today = ql.Date(1, ql.December, 2018)
    ql.Settings.instance().evaluationDate = today
    riskFreeRate = ql.FlatForward(today, r, ql.Actual365Fixed())
    dividendRate = ql.FlatForward(today, 0.0, ql.Actual365Fixed())

    # maturity_date = ql.Date(31, ql.December, 2018)
    maturity_date = today + TimeSteps

    payoff = ql.PlainVanillaPayoff(ql.Option.Call, K)
    exercise = ql.EuropeanExercise(maturity_date)
    european_option = ql.VanillaOption(payoff, exercise)

    # discretization = Reflection  # PartialTrunction, FullTruncation, Reflection, ExactVariance
    heston_process = ql.HestonProcess(ql.YieldTermStructureHandle(riskFreeRate),
                                      ql.YieldTermStructureHandle(dividendRate),
                                      ql.QuoteHandle(ql.SimpleQuote(S0)),  # S0
                                      V0,  # v0
                                      kappa,  # kappa
                                      theta,  # theta
                                      sigma,  # sigma
                                      rho,  # rho
                                      )

    engine = ql.AnalyticHestonEngine(ql.HestonModel(heston_process))
    european_option.setPricingEngine(engine)
    h_price = european_option.NPV()
    return h_price


def heston_VIX_MC(V, NumOfPaths=20000):
    dt = 1. / 365
    T = 30 * dt
    NumSteps = V.shape[0]
    VIX = np.zeros(NumSteps)
    VIX[0] = V[0]
    for i in range(1, NumSteps):
        _, Vs = Simulate_Heston_QuantLib(V0=V[i-1], NumOfAssets=NumOfPaths, TimeSteps=30)
        VIX[i] = 1/T * np.sum(Vs[:, :] * dt, axis=0).mean()
    return VIX


def heston_VIX2(V, kappa, theta):
    T = 30/365
    a = (1 - np.exp(-kappa*T)) / (kappa*T)
    b = theta*(1 - a)
    VIX2 = (a*V + b)*100**2
    return VIX2


def heston_VIX_fwd(VIX):
    return VIX[-1, :].mean()


def heston_VIX_opt(VIX, K, r, t, type='call'):
    if type == 'call':
        payoff = lambda x: np.maximum(x - K, 0)
    elif type == 'put':
        payoff = lambda x: np.maximum(K - x, 0)
    else:
        raise ValueError('Wrong option type.')
    return payoff(VIX).mean()*np.exp(-r*t)


if __name__ == "__main__":
    params = HestonParams()
    strikes_put = np.array([55, 60, 65, 70, 75, 80, 85, 90, 95, 100])
    strikes_call = np.array([100, 105, 110, 115, 120, 125, 130, 135, 140, 145])
    strikes = np.concatenate([strikes_put, strikes_call])
    maturities = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48])
    # S, V = Simulate_Heston_QuantLib(NumOfAssets=60000, TimeSteps=48, dt=1./720)
    np.random.seed(1)

    # VIX2 = heston_VIX2(V, params.kappa, params.theta)
    # Fwd_price = heston_VIX_fwd(np.sqrt(VIX2))

    Fwd = np.zeros(len(maturities))

    for idx, maturity in enumerate(maturities):
        print(f'Calculating maturity: {maturity}')
        S, V = Simulate_Heston_QuantLib(NumOfAssets=60000, TimeSteps=int(maturity*15), dt=1./720)
        Fwd[idx] = heston_VIX_fwd(np.sqrt(heston_VIX2(V, params.kappa, params.theta)))

    np.save('VIX_heston_forwards.npy', Fwd)

    print('end')
