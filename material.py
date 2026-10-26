"""
БЛОК 2 — МАТЕРИАЛЫ.

Класс Material и библиотека предустановленных материалов MATERIALS.
Соответствует БЛОКУ 2 исходного монолита.
"""


class Material:
    def __init__(self, name, sigma_f, sigma_a, sigma_s, nu, density, molar_mass, color):
        self.name       = name
        self.sigma_f    = sigma_f
        self.sigma_a    = sigma_a
        self.sigma_s    = sigma_s
        self.nu         = nu
        self.density    = density
        self.molar_mass = molar_mass
        self.color      = color        # цвет для визуализации

    def sigma_total(self):
        # ИСПРАВЛЕНИЕ: Теперь полное сечение включает деление (sigma_f)
        return self.sigma_f + self.sigma_a + self.sigma_s

    def macro_total(self):
        N = (6.022e23 * self.density) / self.molar_mass
        return N * self.sigma_total() * 1e-24

    def mean_free_path(self):
        return 1.0 / self.macro_total()


MATERIALS = {
    "U-235 (чистый)":  Material("U-235",  sigma_f=1.65,  sigma_a=2.71,  sigma_s=8.9,  nu=2.43, density=19.1, molar_mass=235, color="#F5A623"),
    "U-235 (3.5%)":    Material("U-235 обогащ.", sigma_f=0.06, sigma_a=0.45, sigma_s=9.5, nu=2.43, density=10.5, molar_mass=238, color="#E8742A"),
    "Вода (H₂O)":      Material("H₂O",    sigma_f=0,     sigma_a=0.66,  sigma_s=49,   nu=0,    density=1.0,  molar_mass=18,  color="#4FC3F7"),
    "Графит":          Material("Графит",  sigma_f=0,     sigma_a=0.004, sigma_s=4.7,  nu=0,    density=1.7,  molar_mass=12,  color="#78909C"),
}
