class APAction:
    """
    Clase auxiliar para interpretar la acción del agente a nivel de Access Point (AP).

    Convención de acción:
    action = [ch_0, pwr_0, ch_1, pwr_1, ..., ch_N, pwr_N]

    Cada par (ch_i, pwr_i) corresponde al AP i.
    Los valores son índices discretos, no valores físicos.
    """

    def __init__(self, action, num_aps):
        self.raw = action.reshape(num_aps, 2)

    def channel(self, ap_idx):
        """
        Devuelve el índice del canal seleccionado para el AP indicado.
        """
        return self.raw[ap_idx, 0]

    def power(self, ap_idx):
        """
        Devuelve el índice de la potencia de transmisión seleccionada
        para el AP indicado.
        """
        return self.raw[ap_idx, 1]
