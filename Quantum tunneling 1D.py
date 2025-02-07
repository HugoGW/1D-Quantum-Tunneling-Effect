import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from scipy.constants import h, hbar, e, m_e

# Définition des constantes et paramètres
DeuxPi = 2.0 * np.pi
L = 5.0e-9  # dimension de la boîte quantique en mètres
Nx = 2000  # nombre de points spatiaux
xmin, xmax = 0.0, L
dx = (xmax - xmin) / Nx
x = np.arange(0.0, L, dx)
Nt = 1e-12  # Durée totale en secondes
a2 = 0.1
dt = a2 * 2 * m_e * dx**2 / hbar
a3 = e * dt / hbar

# Paramètres du paquet d'onde initial
x0 = x[int(Nx / 4)]  # position initiale du paquet
sigma = 2.0e-10  # largeur du paquet en m
Lambda = 1.5e-10  # longueur d'onde de de Broglie de l'électron en m
Ec = (h / Lambda)**2 / (2 * m_e * e)  # énergie cinétique en eV
print(Ec)

# Définition du potentiel
U0 = 80  # en eV
a = 7e-11  # largeur de la barrière en mètres (modifiable)
center = L / 2  # position centrale de la barrière
barrier_start = center - a / 2
barrier_end = center + a / 2

# Construction du potentiel en fonction de la position de la barrière
U = np.zeros(Nx)
U[(x >= barrier_start) & (x <= barrier_end)] = U0  # barrière de potentiel

# Indices de début et de fin de la barrière
barrier_start_index = int(barrier_start / dx)
barrier_end_index = int(barrier_end / dx)

# Initialisation des buffers de calcul aux conditions initiales
Psi_Real = np.exp(-0.5 * ((x - x0) / sigma)**2) * np.cos(DeuxPi * (x - x0) / Lambda)
Psi_Imag = np.exp(-0.5 * ((x - x0) / sigma)**2) * np.sin(DeuxPi * (x - x0) / Lambda)
Psi_Prob = Psi_Real**2 + Psi_Imag**2  # densité de probabilité

# Création de la figure et des axes pour l'animation
fig, ax1 = plt.subplots(figsize=(10,8))
line_prob, = ax1.plot([], [], 'darkred')  # On initialise la courbe sans la tracer
ax1.set_xlim(0, L * 1.e9)
ax1.set_ylim(0.00001, 1.1*max(Psi_Prob/np.sum(Psi_Prob)))
ax1.set_xlabel('x [nm]')
ax1.set_ylabel(r'Detection probability density [$m^{-1}$]')
ax1.grid(True)

ax2 = ax1.twinx()
ax2.plot(x * 1.e9, U, 'royalblue')
ax2.set_ylabel('U [eV]')
ax2.set_ylim(0.2, 90)

# Adding text labels for Psi^2 to the left and right of the barrier
left_prob_text = ax1.text(0.1, 0.91, '', transform=ax1.transAxes, color='black', fontsize=14)
right_prob_text = ax1.text(0.55, 0.91, '', transform=ax1.transAxes, color='black', fontsize=14)
T_text = ax1.text(0.77, 0.91, '', transform=ax1.transAxes, color='black', fontsize=14)

# Texte pour afficher le temps écoulé
time_text = ax1.text(0.77, 0.81, '', transform=ax1.transAxes, color='blue', fontsize=14)

K = np.sqrt(2*m_e*(U0-Ec)*e)/hbar
T = 1/((U0**2/(4*Ec*(U0-Ec))*np.sinh(K*a)**2)+1)

# Variable pour suivre le temps
current_time = 0.0

# Fonction de mise à jour pour chaque frame de l'animation
def update(frame):
    global Psi_Real, Psi_Imag, Psi_Prob, current_time

    # Arrêter l'animation si le temps dépasse Nt/1000
    if current_time > Nt/1000:
        ani.event_source.stop()
        plt.close(fig)
        return

    for _ in range(60):
        Psi_Real[1:-1] = Psi_Real[1:-1] - a2 * (Psi_Imag[2:] - 2 * Psi_Imag[1:-1] + Psi_Imag[:-2]) + a3 * U[1:-1] * Psi_Imag[1:-1]
        Psi_Imag[1:-1] = Psi_Imag[1:-1] + a2 * (Psi_Real[2:] - 2 * Psi_Real[1:-1] + Psi_Real[:-2]) - a3 * U[1:-1] * Psi_Real[1:-1]
        Psi_Prob[1:-1] = Psi_Real[1:-1]**2 + Psi_Imag[1:-1]**2
        current_time += dt

    # Mise à jour de la densité de probabilité pour l'animation
    line_prob.set_data(x * 1.e9, Psi_Prob/np.sum(Psi_Prob))

    # Calcul des probabilités à gauche et à droite de la barrière
    left_side_prob = np.sum(Psi_Prob[:barrier_start_index]) / np.sum(Psi_Prob)
    right_side_prob = np.sum(Psi_Prob[barrier_end_index:]) / np.sum(Psi_Prob)

    # Mise à jour des textes des probabilités
    left_prob_text.set_text(f'$\Psi^2_L = $ {left_side_prob:.2e}')
    right_prob_text.set_text(f'$\Psi^2_R = $ {right_side_prob:.2e}')
    T_text.set_text(f'$T =$ {T:.2e}')

    # Mise à jour du texte du temps écoulé
    time_text.set_text(f'Time: {current_time*1e15:.2f} fs')  # Conversion en femtosecondes

    return line_prob, left_prob_text, right_prob_text, T_text, time_text

# Création de l'animation
ani = FuncAnimation(fig, update, frames=range(0, int(Nt/dt)), interval=1, blit=True)
plt.tight_layout()

# # Définir le chemin de sauvegarde
# output_path = r"P:\\Cours Physique - Universités\\Cours fac UBFC\\M2\\S9\\Free Numerical Project\\alpha version\\QT_1D.mp4"

# # Création du writer pour sauvegarder en MP4
# writer = FFMpegWriter(fps=30, metadata=dict(artist='Hugo Alexandre'), bitrate=1800)

# # Enregistrement de l'animation
# with writer.saving(fig, output_path, dpi=200):
#     for frame in range(0, int(Nt/dt)):
#         update(frame)  # Appel de la fonction update pour chaque frame
#         writer.grab_frame()

plt.show()
