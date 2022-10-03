# LongTermSpectralFlatness, SpectralSlope have some unknown issuses,
# the results are different from the matlab's results.
import torch
from speechbrain.processing.features import STFT, spectral_magnitude


class SpectralEntropy(torch.nn.Module):
    """
    Spectral entropy has been used successfully in voiced/unvoiced
    decisions for automatic speech recognition. Because entropy is
    a measure of disorder, regions of voiced speech have lower
    entropy compared to regions of unvoiced speech.

    Arguments
    ---------
    sample_rate : int
        Sample rate of the input audio signal (e.g 16000).
    win_length : float
        Length (in ms) of the sliding window used to compute the STFT.
    hop_length : float
        Length (in ms) of the hope of the sliding window used to compute
        the STFT.
    n_fft : int
        Number of fft point of the STFT. It defines the frequency resolution
        (n_fft should be <= than win_len).
    window_fn : function
        A function that takes an integer (number of samples) and outputs a
        tensor to be multiplied with each window before fft.
    spectrum_type : str
        Spectrum type, specified as "power" or "magnitude":
        "power": The spectral entropy is calculated for the one-sided power spectrum.
        "magnitude": The spectral entropy is calculated for the one-sided magnitude spectrum.
    normalized_entropy : bool
        If True, divide by log(bins.size) to normalize the spectral entropy
        between 0 and 1.

    Example
    -------
    >>> import torch
    >>> compute_stft = STFT(sample_rate=16000)
    >>> compute_feat = SpectralEntropy(sample_rate=16000)
    >>> inputs = torch.randn([10, 16000])
    >>> features = compute_feat(inputs)
    >>> features.shape
    ... torch.Size([10, 101, 1])
    >>> spectr = compute_stft(inputs)
    >>> features = compute_feat(spectr)
    >>> features.shape
    ... torch.Size([10, 101, 1])
    """

    def __init__(
        self,
        sample_rate,
        win_length=25,
        hop_length=10,
        n_fft=400,
        window_fn=torch.hamming_window,
        spectrum_type="power",
        normalized_entropy=False,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.spectrum_type = spectrum_type
        self.normalized_entropy = normalized_entropy
        self.window = window_fn(win_length)
        self.stft = STFT(
            sample_rate=sample_rate,
            win_length=win_length,
            hop_length=hop_length,
            n_fft=n_fft,
            window_fn=window_fn,
        )

    def forward(self, x):
        """
        x : tensor
            The name-value arguments apply if x is a batch of time-domain
            signals to transform. The tensor must have the format
            (batch, time_step). If x is a batch of frequency-domain signal,
            name-value arguments are ignored. The tensor must ave the format
            (batch, time_step, n_fft/2 + 1, 2).
        """

        if x.ndim == 2:
            x -= x.mean(axis=1, keepdim=True)
            x = self.stft(x)

        # Compute power spectral density(PSD)
        if self.spectrum_type == "magnitude":
            spectr = spectral_magnitude(x, 0.5)
        else:
            spectr = spectral_magnitude(x)

        # Compute the cross spectral density where `Pxy` has units of V**2/Hz
        # https://github.com/scipy/scipy/blob/2e5883ef7af4f5ed4a5b80a1759a45e43163bf3f/scipy/signal/_spectral_py.py#L1840
        psd = spectr / self.sample_rate * (self.window**2).sum()

        # Last point is unpaired Nyquist freq point, don't double.
        # Then average over windows.
        psd[..., 1:-1] *= 2
        psd = psd.mean(dim=-1, keepdim=True)

        # Normalize to be viewed as a probability density function (PDF)
        psd_norm = psd / psd.sum(dim=1, keepdim=True)

        entropy = - (psd_norm * psd_norm.log())

        if self.normalized_entropy:
            entropy /= torch.log(torch.tensor([psd_norm.shape[-1]]))

        return entropy


class SpectralCentroid(torch.nn.Module):
    """
    The spectral centroid represents the "center of gravity" of the spectrum.
    It is used as an indication of brightness and is commonly used in music
    analysis and genre classification.

    Arguments
    ---------
    sample_rate : int
        Sample rate of the input audio signal (e.g 16000).
    win_length : float
        Length (in ms) of the sliding window used to compute the STFT.
    hop_length : float
        Length (in ms) of the hope of the sliding window used to compute
        the STFT.
    n_fft : int
        Number of fft point of the STFT. It defines the frequency resolution
        (n_fft should be <= than win_len).
    window_fn : function
        A function that takes an integer (number of samples) and outputs a
        tensor to be multiplied with each window before fft.
    spectrum_type : str
        Spectrum type, specified as "power" or "magnitude":
        "power": The spectral entropy is calculated for the one-sided power spectrum.
        "magnitude": The spectral entropy is calculated for the one-sided magnitude spectrum.

    Example
    -------
    >>> import torch
    >>> compute_stft = STFT(sample_rate=16000)
    >>> compute_feat = SpectralCentroid(sample_rate=16000)
    >>> inputs = torch.randn([10, 16000])
    >>> features = compute_feat(inputs)
    >>> features.shape
    ... torch.Size([10, 101, 1])
    >>> spectr = compute_stft(inputs)
    >>> features = compute_feat(spectr)
    >>> features.shape
    ... torch.Size([10, 101, 1])
    """

    def __init__(
        self,
        sample_rate,
        win_length=25,
        hop_length=10,
        n_fft=400,
        window_fn=torch.hamming_window,
        spectrum_type="power",
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.spectrum_type = spectrum_type
        self.stft = STFT(
            sample_rate=sample_rate,
            win_length=win_length,
            hop_length=hop_length,
            n_fft=n_fft,
            window_fn=window_fn,
        )

    def forward(self, x):
        """
        x : tensor
            The name-value arguments apply if x is a batch of time-domain
            signals to transform. The tensor must have the format
            (batch, time_step). If x is a batch of frequency-domain signal,
            name-value arguments are ignored. The tensor must ave the format
            (batch, time_step, n_fft/2 + 1, 2).
        """

        if x.ndim == 2:
            x = self.stft(x)

        if self.spectrum_type == "magnitude":
            spectr = spectral_magnitude(x, 0.5)
        else:
            spectr = spectral_magnitude(x)

        n_fft = (x.shape[2] - 1) * 2
        freqs = torch.fft.rfftfreq(n_fft, d=1./self.sample_rate, device=x.device)
        centroid = (
            (spectr * freqs).sum(dim=-1, keepdim=True)
            / spectr.sum(dim=-1, keepdim=True)
        )

        return centroid


class SpectralSpread(torch.nn.Module):
    """
    The spectral spread represents the "instantaneous bandwidth" of
    the spectrum. It is used as an indication of the dominance of
    a tone. For example, the spread increases as the tones diverge
    and decreases as the tones converge.

    Arguments
    ---------
    sample_rate : int
        Sample rate of the input audio signal (e.g 16000).
    win_length : float
        Length (in ms) of the sliding window used to compute the STFT.
    hop_length : float
        Length (in ms) of the hope of the sliding window used to compute
        the STFT.
    n_fft : int
        Number of fft point of the STFT. It defines the frequency resolution
        (n_fft should be <= than win_len).
    window_fn : function
        A function that takes an integer (number of samples) and outputs a
        tensor to be multiplied with each window before fft.
    spectrum_type : str
        Spectrum type, specified as "power" or "magnitude":
        "power": The spectral entropy is calculated for the one-sided power spectrum.
        "magnitude": The spectral entropy is calculated for the one-sided magnitude spectrum.

    Example
    -------
    >>> import torch
    >>> compute_stft = STFT(sample_rate=16000)
    >>> compute_feat = SpectralSpread(sample_rate=16000)
    >>> inputs = torch.randn([10, 16000])
    >>> features = compute_feat(inputs)
    >>> features.shape
    ... torch.Size([10, 101, 1])
    >>> spectr = compute_stft(inputs)
    >>> features = compute_feat(spectr)
    >>> features.shape
    ... torch.Size([10, 101, 1])
    """

    def __init__(
        self,
        sample_rate,
        win_length=25,
        hop_length=10,
        n_fft=400,
        window_fn=torch.hamming_window,
        spectrum_type="power",
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.spectrum_type = spectrum_type
        self.stft = STFT(
            sample_rate=sample_rate,
            win_length=win_length,
            hop_length=hop_length,
            n_fft=n_fft,
            window_fn=window_fn,
        )
        self.centroid = SpectralCentroid(
            sample_rate=sample_rate,
            win_length=win_length,
            hop_length=hop_length,
            n_fft=n_fft,
            window_fn=window_fn,
            spectrum_type=spectrum_type,
        )

    def forward(self, x):
        """
        x : tensor
            The name-value arguments apply if x is a batch of time-domain
            signals to transform. The tensor must have the format
            (batch, time_step). If x is a batch of frequency-domain signal,
            name-value arguments are ignored. The tensor must ave the format
            (batch, time_step, n_fft/2 + 1, 2).
        """
        u_1 = self.centroid(x)

        if x.ndim == 2:
            x = self.stft(x)

        if self.spectrum_type == "magnitude":
            spectr = spectral_magnitude(x, 0.5)
        else:
            spectr = spectral_magnitude(x)

        n_fft = (x.shape[2] - 1) * 2
        freqs = torch.fft.rfftfreq(n_fft, d=1./self.sample_rate, device=x.device)
        freqs = (freqs[None, None, :] - u_1).pow(2)
        spread = (
            (spectr * freqs).sum(dim=-1, keepdim=True)
            / spectr.sum(dim=-1, keepdim=True)
        ) ** 0.5

        return spread


class SpectralSkewness(torch.nn.Module):
    """
    The spectral skewness measures symmetry around the centroid. In phonetics,
    spectral skewness is often referred to as spectral tilt and is used with
    other spectral moments to distinguish the place of articulation

    Arguments
    ---------
    sample_rate : int
        Sample rate of the input audio signal (e.g 16000).
    win_length : float
        Length (in ms) of the sliding window used to compute the STFT.
    hop_length : float
        Length (in ms) of the hope of the sliding window used to compute
        the STFT.
    n_fft : int
        Number of fft point of the STFT. It defines the frequency resolution
        (n_fft should be <= than win_len).
    window_fn : function
        A function that takes an integer (number of samples) and outputs a
        tensor to be multiplied with each window before fft.
    spectrum_type : str
        Spectrum type, specified as "power" or "magnitude":
        "power": The spectral entropy is calculated for the one-sided power spectrum.
        "magnitude": The spectral entropy is calculated for the one-sided magnitude spectrum.

    Example
    -------
    >>> import torch
    >>> compute_stft = STFT(sample_rate=16000)
    >>> compute_feat = SpectralSkewness(sample_rate=16000)
    >>> inputs = torch.randn([10, 16000])
    >>> features = compute_feat(inputs)
    >>> features.shape
    ... torch.Size([10, 101, 1])
    >>> spectr = compute_stft(inputs)
    >>> features = compute_feat(spectr)
    >>> features.shape
    ... torch.Size([10, 101, 1])
    """

    def __init__(
        self,
        sample_rate,
        win_length=25,
        hop_length=10,
        n_fft=400,
        window_fn=torch.hamming_window,
        spectrum_type="power",
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.spectrum_type = spectrum_type
        self.stft = STFT(
            sample_rate=sample_rate,
            win_length=win_length,
            hop_length=hop_length,
            n_fft=n_fft,
            window_fn=window_fn,
        )
        self.centroid = SpectralCentroid(
            sample_rate=sample_rate,
            win_length=win_length,
            hop_length=hop_length,
            n_fft=n_fft,
            window_fn=window_fn,
            spectrum_type=spectrum_type,
        )
        self.spread = SpectralSpread(
            sample_rate=sample_rate,
            win_length=win_length,
            hop_length=hop_length,
            n_fft=n_fft,
            window_fn=window_fn,
            spectrum_type=spectrum_type,
        )

    def forward(self, x):
        """
        x : tensor
            The name-value arguments apply if x is a batch of time-domain
            signals to transform. The tensor must have the format
            (batch, time_step). If x is a batch of frequency-domain signal,
            name-value arguments are ignored. The tensor must ave the format
            (batch, time_step, n_fft/2 + 1, 2).
        """

        u_1 = self.centroid(x)
        u_2 = self.spread(x)

        if x.ndim == 2:
            x = self.stft(x)

        if self.spectrum_type == "magnitude":
            spectr = spectral_magnitude(x, 0.5)
        else:
            spectr = spectral_magnitude(x)

        n_fft = (x.shape[2] - 1) * 2
        freqs = torch.fft.rfftfreq(n_fft, d=1./self.sample_rate, device=x.device)
        freqs = (freqs[None, None, :] - u_1).pow(3)
        skewness = (
            (spectr * freqs).sum(dim=-1, keepdim=True)
            / (u_2.pow(3) * spectr.sum(dim=-1, keepdim=True))
        )

        return skewness


class SpectralKurtosis(torch.nn.Module):
    """
    The spectral kurtosis measures the flatness, or non-Gaussianity,
    of the spectrum around its centroid. Conversely, it is used to
    indicate the peakiness of a spectrum. For example, as the white
    noise is increased on the speech signal, the kurtosis decreases,
    indicating a less peaky spectrum.

    Arguments
    ---------
    sample_rate : int
        Sample rate of the input audio signal (e.g 16000).
    win_length : float
        Length (in ms) of the sliding window used to compute the STFT.
    hop_length : float
        Length (in ms) of the hope of the sliding window used to compute
        the STFT.
    n_fft : int
        Number of fft point of the STFT. It defines the frequency resolution
        (n_fft should be <= than win_len).
    window_fn : function
        A function that takes an integer (number of samples) and outputs a
        tensor to be multiplied with each window before fft.
    spectrum_type : str
        Spectrum type, specified as "power" or "magnitude":
        "power": The spectral entropy is calculated for the one-sided power spectrum.
        "magnitude": The spectral entropy is calculated for the one-sided magnitude spectrum.

    Example
    -------
    >>> import torch
    >>> compute_stft = STFT(sample_rate=16000)
    >>> compute_feat = SpectralKurtosis(sample_rate=16000)
    >>> inputs = torch.randn([10, 16000])
    >>> features = compute_feat(inputs)
    >>> features.shape
    ... torch.Size([10, 101, 1])
    >>> spectr = compute_stft(inputs)
    >>> features = compute_feat(spectr)
    >>> features.shape
    ... torch.Size([10, 101, 1])
    """

    def __init__(
        self,
        sample_rate,
        win_length=25,
        hop_length=10,
        n_fft=400,
        window_fn=torch.hamming_window,
        spectrum_type="power",
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.spectrum_type = spectrum_type
        self.stft = STFT(
            sample_rate=sample_rate,
            win_length=win_length,
            hop_length=hop_length,
            n_fft=n_fft,
            window_fn=window_fn,
        )
        self.centroid = SpectralCentroid(
            sample_rate=sample_rate,
            win_length=win_length,
            hop_length=hop_length,
            n_fft=n_fft,
            window_fn=window_fn,
            spectrum_type=spectrum_type,
        )
        self.spread = SpectralSpread(
            sample_rate=sample_rate,
            win_length=win_length,
            hop_length=hop_length,
            n_fft=n_fft,
            window_fn=window_fn,
            spectrum_type=spectrum_type,
        )

    def forward(self, x):
        """
        x : tensor
            The name-value arguments apply if x is a batch of time-domain
            signals to transform. The tensor must have the format
            (batch, time_step). If x is a batch of frequency-domain signal,
            name-value arguments are ignored. The tensor must ave the format
            (batch, time_step, n_fft/2 + 1, 2).
        """

        u_1 = self.centroid(x)
        u_2 = self.spread(x)

        if x.ndim == 2:
            x = self.stft(x)

        if self.spectrum_type == "magnitude":
            spectr = spectral_magnitude(x, 0.5)
        else:
            spectr = spectral_magnitude(x)

        n_fft = (x.shape[2] - 1) * 2
        freqs = torch.fft.rfftfreq(n_fft, d=1./self.sample_rate, device=x.device)
        freqs = (freqs[None, None, :] - u_1).pow(4)
        kurtosis = (
            (spectr * freqs).sum(dim=-1, keepdim=True)
            / (u_2.pow(4) * spectr.sum(dim=-1, keepdim=True))
        )

        return kurtosis


class SpectralRolloffPoint(torch.nn.Module):
    """
    Alternative implementation of librosa.feature.spectral_rolloff.
    The spectral rolloff point measures the bandwidth of the audio signal
    by determining the frequency bin under which a given percentage of
    the total energy exists. It has been used to distinguish between
    voiced and unvoiced speech, speech/music discrimination, etc.

    Arguments
    ---------
    sample_rate : int
        Sample rate of the input audio signal (e.g 16000).
    win_length : float
        Length (in ms) of the sliding window used to compute the STFT.
    hop_length : float
        Length (in ms) of the hope of the sliding window used to compute
        the STFT.
    n_fft : int
        Number of fft point of the STFT. It defines the frequency resolution
        (n_fft should be <= than win_len).
    window_fn : function
        A function that takes an integer (number of samples) and outputs a
        tensor to be multiplied with each window before fft.
    spectrum_type : str
        Spectrum type, specified as "power" or "magnitude":
        "power": The spectral entropy is calculated for the one-sided power spectrum.
        "magnitude": The spectral entropy is calculated for the one-sided magnitude spectrum.
    roll_threshold : float
        The threshold of rolloff point, specified as a scalar between zero and one.
        Usually 0.85 or 0.95, default to 0.95.

    Example
    -------
    >>> import torch
    >>> compute_stft = STFT(sample_rate=16000)
    >>> compute_feat = SpectralRolloffPoint(sample_rate=16000)
    >>> inputs = torch.randn([10, 16000])
    >>> features = compute_feat(inputs)
    >>> features.shape
    ... torch.Size([10, 101, 1])
    >>> spectr = compute_stft(inputs)
    >>> features = compute_feat(spectr)
    >>> features.shape
    ... torch.Size([10, 101, 1])
    """

    def __init__(
        self,
        sample_rate,
        win_length=25,
        hop_length=10,
        n_fft=400,
        window_fn=torch.hamming_window,
        spectrum_type="power",
        roll_threshold=0.95,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.spectrum_type = spectrum_type
        self.threshold = roll_threshold
        self.stft = STFT(
            sample_rate=sample_rate,
            win_length=win_length,
            hop_length=hop_length,
            n_fft=n_fft,
            window_fn=window_fn,
        )

    def forward(self, x):
        """
        x : tensor
            The name-value arguments apply if x is a batch of time-domain
            signals to transform. The tensor must have the format
            (batch, time_step). If x is a batch of frequency-domain signal,
            name-value arguments are ignored. The tensor must ave the format
            (batch, time_step, n_fft/2 + 1, 2).
        """

        if not 0.0 < self.threshold < 1.0:
            raise ValueError("roll_threshold must specift as a scalar between 0 and 1.")

        if x.ndim == 2:
            x = self.stft(x)

        if self.spectrum_type == "magnitude":
            spectr = spectral_magnitude(x, 0.5)
        else:
            spectr = spectral_magnitude(x)

        n_fft = (x.shape[2] - 1) * 2
        freqs = torch.fft.rfftfreq(n_fft, d=1./self.sample_rate, device=x.device)

        total_energy = torch.cumsum(spectr, dim=-1)
        threshold = self.threshold * total_energy[..., -1]
        idx = torch.where(total_energy > threshold[..., None], 1., float("nan"))

        rolloff = torch.topk(idx * freqs, 1, largest=False, dim=-1).values

        return rolloff


class SpectralCrest(torch.nn.Module):
    """
    Spectral crest is an indication of the peakiness of the spectrum.
    A higher spectral crest indicates more tonality, while a lower
    spectral crest indicates more noise.

    Arguments
    ---------
    sample_rate : int
        Sample rate of the input audio signal (e.g 16000).
    win_length : float
        Length (in ms) of the sliding window used to compute the STFT.
    hop_length : float
        Length (in ms) of the hope of the sliding window used to compute
        the STFT.
    n_fft : int
        Number of fft point of the STFT. It defines the frequency resolution
        (n_fft should be <= than win_len).
    window_fn : function
        A function that takes an integer (number of samples) and outputs a
        tensor to be multiplied with each window before fft.
    spectrum_type : str
        Spectrum type, specified as "power" or "magnitude":
        "power": The spectral entropy is calculated for the one-sided power spectrum.
        "magnitude": The spectral entropy is calculated for the one-sided magnitude spectrum.

    Example
    -------
    >>> import torch
    >>> compute_stft = STFT(sample_rate=16000)
    >>> compute_feat = SpectralCrest(sample_rate=16000)
    >>> inputs = torch.randn([10, 16000])
    >>> features = compute_feat(inputs)
    >>> features.shape
    ... torch.Size([10, 101, 1])
    >>> spectr = compute_stft(inputs)
    >>> features = compute_feat(spectr)
    >>> features.shape
    ... torch.Size([10, 101, 1])
    """

    def __init__(
        self,
        sample_rate,
        win_length=25,
        hop_length=10,
        n_fft=400,
        window_fn=torch.hamming_window,
        spectrum_type="power",
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.spectrum_type = spectrum_type
        self.stft = STFT(
            sample_rate=sample_rate,
            win_length=win_length,
            hop_length=hop_length,
            n_fft=n_fft,
            window_fn=window_fn,
        )

    def forward(self, x):
        """
        x : tensor
            The name-value arguments apply if x is a batch of time-domain
            signals to transform. The tensor must have the format
            (batch, time_step). If x is a batch of frequency-domain signal,
            name-value arguments are ignored. The tensor must ave the format
            (batch, time_step, n_fft/2 + 1, 2).
        """

        if x.ndim == 2:
            x = self.stft(x)

        if self.spectrum_type == "magnitude":
            spectr = spectral_magnitude(x, 0.5)
        else:
            spectr = spectral_magnitude(x)

        crest = (
            spectr.max(dim=-1, keepdim=True).values
            / spectr.mean(dim=-1, keepdim=True)
        )

        return crest


class SpectralFlux(torch.nn.Module):
    """
    Spectral flux is a measure of the variability of the spectrum over
    time. It is popularly used in onset detection and audio segmentation.

    Arguments
    ---------
    sample_rate : int
        Sample rate of the input audio signal (e.g 16000).
    win_length : float
        Length (in ms) of the sliding window used to compute the STFT.
    hop_length : float
        Length (in ms) of the hope of the sliding window used to compute
        the STFT.
    n_fft : int
        Number of fft point of the STFT. It defines the frequency resolution
        (n_fft should be <= than win_len).
    window_fn : function
        A function that takes an integer (number of samples) and outputs a
        tensor to be multiplied with each window before fft.
    spectrum_type : str
        Spectrum type, specified as "power" or "magnitude":
        "power": The spectral entropy is calculated for the one-sided power spectrum.
        "magnitude": The spectral entropy is calculated for the one-sided magnitude spectrum.
    norm_type : int
        Norm type used to calculate flux, specified as 2 or 1. Default to 2.

    Example
    -------
    >>> import torch
    >>> compute_stft = STFT(sample_rate=16000)
    >>> compute_feat = SpectralFlux(sample_rate=16000)
    >>> inputs = torch.randn([10, 16000])
    >>> features = compute_feat(inputs)
    >>> features.shape
    ... torch.Size([10, 101, 1])
    >>> spectr = compute_stft(inputs)
    >>> features = compute_feat(spectr)
    >>> features.shape
    ... torch.Size([10, 101, 1])
    """

    def __init__(
        self,
        sample_rate,
        win_length=25,
        hop_length=10,
        n_fft=400,
        window_fn=torch.hamming_window,
        spectrum_type="power",
        norm_type=2,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.spectrum_type = spectrum_type
        self.p = norm_type
        self.stft = STFT(
            sample_rate=sample_rate,
            win_length=win_length,
            hop_length=hop_length,
            n_fft=n_fft,
            window_fn=window_fn,
            normalized_stft=True,
        )

    def forward(self, x):
        """
        x : tensor
            The name-value arguments apply if x is a batch of time-domain
            signals to transform. The tensor must have the format
            (batch, time_step). If x is a batch of frequency-domain signal,
            name-value arguments are ignored. The tensor must ave the format
            (batch, time_step, n_fft/2 + 1, 2).
        """

        if x.ndim == 2:
            x = self.stft(x)

        if self.spectrum_type == "magnitude":
            spectr = spectral_magnitude(x, 0.5)
        else:
            spectr = spectral_magnitude(x)

        flux = torch.sum(
            (spectr[:, 1:, :] - spectr[:, :-1, :]).abs().pow(self.p),
            dim=-1,
            keepdim=True,
        ) ** (1. / self.p)

        flux /= spectr.shape[-1]
        offset = torch.zeros((flux.shape[0], 1, flux.shape[2]), device=x.device)
        flux = torch.concat((offset, flux), dim=1)

        return flux


class SpectralSlope(torch.nn.Module):
    """
    WIP
    <intro>

    Arguments
    ---------
    sample_rate : int
        Sample rate of the input audio signal (e.g 16000).
    win_length : float
        Length (in ms) of the sliding window used to compute the STFT.
    hop_length : float
        Length (in ms) of the hope of the sliding window used to compute
        the STFT.
    n_fft : int
        Number of fft point of the STFT. It defines the frequency resolution
        (n_fft should be <= than win_len).
    window_fn : function
        A function that takes an integer (number of samples) and outputs a
        tensor to be multiplied with each window before fft.
    spectrum_type : str
        Spectrum type, specified as "power" or "magnitude":
        "power": The spectral entropy is calculated for the one-sided power spectrum.
        "magnitude": The spectral entropy is calculated for the one-sided magnitude spectrum.

    Example
    -------
    >>> import torch
    >>> compute_stft = STFT(sample_rate=16000)
    >>> compute_feat = SpectralSlope(sample_rate=16000)
    >>> inputs = torch.randn([10, 16000])
    >>> features = compute_feat(inputs)
    >>> features.shape
    ... torch.Size([10, 101, 1])
    >>> spectr = compute_stft(inputs)
    >>> features = compute_feat(spectr)
    >>> features.shape
    ... torch.Size([10, 101, 1])
    """

    def __init__(
        self,
        sample_rate,
        win_length=25,
        hop_length=10,
        n_fft=400,
        window_fn=torch.hamming_window,
        spectrum_type="power",
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.spectrum_type = spectrum_type
        self.stft = STFT(
            sample_rate=sample_rate,
            win_length=win_length,
            hop_length=hop_length,
            n_fft=n_fft,
            window_fn=window_fn,
        )

    def forward(self, x):
        """
        x : tensor
            The name-value arguments apply if x is a batch of time-domain
            signals to transform. The tensor must have the format
            (batch, time_step). If x is a batch of frequency-domain signal,
            name-value arguments are ignored. The tensor must ave the format
            (batch, time_step, n_fft/2 + 1, 2).
        """

        if x.ndim == 2:
            x = self.stft(x)

        if self.spectrum_type == "magnitude":
            spectr = spectral_magnitude(x, 0.5)
        else:
            spectr = spectral_magnitude(x)

        n_fft = (x.shape[2] - 1) * 2
        freqs = torch.fft.rfftfreq(n_fft, d=1./self.sample_rate, device=x.device)

        freqs -= freqs.mean()
        spectr -= spectr.mean(dim=-1, keepdim=True)

        slope = (
            (freqs[None, None, :] * spectr).sum(dim=-1, keepdim=True)
            / freqs.pow(2).sum()
        )

        return slope


class SpectralFlatness(torch.nn.Module):
    """
    Spectral flatness is a measure to quantify how much noise-like a sound is.
    A high spectral flatness (closer to 1.0) indicates the spectrum is similar
    to white noise. It is often converted to decibel.

    Arguments
    ---------
    sample_rate : int
        Sample rate of the input audio signal (e.g 16000).
    win_length : float
        Length (in ms) of the sliding window used to compute the STFT.
    hop_length : float
        Length (in ms) of the hope of the sliding window used to compute
        the STFT.
    n_fft : int
        Number of fft point of the STFT. It defines the frequency resolution
        (n_fft should be <= than win_len).
    window_fn : function
        A function that takes an integer (number of samples) and outputs a
        tensor to be multiplied with each window before fft.
    spectrum_type : str
        Spectrum type, specified as "power" or "magnitude":
        "power": The spectral entropy is calculated for the one-sided power spectrum.
        "magnitude": The spectral entropy is calculated for the one-sided magnitude spectrum.

    Example
    -------
    >>> import torch
    >>> compute_stft = STFT(sample_rate=16000)
    >>> compute_feat = SpectralFlatness(sample_rate=16000)
    >>> inputs = torch.randn([10, 16000])
    >>> features = compute_feat(inputs)
    >>> features.shape
    ... torch.Size([10, 101, 1])
    >>> spectr = compute_stft(inputs)
    >>> features = compute_feat(spectr)
    >>> features.shape
    ... torch.Size([10, 101, 1])
    """

    def __init__(
        self,
        sample_rate,
        win_length=25,
        hop_length=10,
        n_fft=400,
        window_fn=torch.hamming_window,
        spectrum_type="power",
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.spectrum_type = spectrum_type
        self.eps = 1e-3
        self.stft = STFT(
            sample_rate=sample_rate,
            win_length=win_length,
            hop_length=hop_length,
            n_fft=n_fft,
            window_fn=window_fn,
        )

    def forward(self, x):
        """
        x : tensor
            The name-value arguments apply if x is a batch of time-domain
            signals to transform. The tensor must have the format
            (batch, time_step). If x is a batch of frequency-domain signal,
            name-value arguments are ignored. The tensor must ave the format
            (batch, time_step, n_fft/2 + 1, 2).
        """

        if x.ndim == 2:
            x = self.stft(x)

        if self.spectrum_type == "magnitude":
            spectr = spectral_magnitude(x, 0.5)
        else:
            spectr = spectral_magnitude(x)

        geometric_mean = (spectr + self.eps).log().mean(dim=-1, keepdim=True).exp() - self.eps
        arithmetic_mean = spectr.mean(dim=-1, keepdim=True)
        flatness = -10. * (self.eps + geometric_mean / arithmetic_mean).log10()

        return flatness


class LongTermSpectralFlatness(torch.nn.Module):
    """
    WIP
    http://link.springer.com/article/10.1186/1687-4722-2013-21

    Arguments
    ---------
    sample_rate : int
        Sample rate of the input audio signal (e.g 16000).
    win_length : float
        Length (in ms) of the sliding window used to compute the STFT.
    hop_length : float
        Length (in ms) of the hope of the sliding window used to compute
        the STFT.
    n_fft : int
        Number of fft point of the STFT. It defines the frequency resolution
        (n_fft should be <= than win_len).
    window_fn : function
        A function that takes an integer (number of samples) and outputs a
        tensor to be multiplied with each window before fft.
    m : int
        The consecutive frames used to average spectral estimates, default to 10.
    r : int
        Long-term window length, default to 30.
    freq_range : tuple
        Frequency range in Hz, specified as a two-element integer in the
        range like (0, 16000). Default to None.

    Example
    -------
    >>> import torch
    >>> compute_stft = STFT(sample_rate=16000)
    >>> compute_feat = LongTermSpectralFlatness(sample_rate=16000)
    >>> inputs = torch.randn([10, 16000])
    >>> features = compute_feat(inputs)
    >>> features.shape
    ... torch.Size([10, 101, 1])
    >>> spectr = compute_stft(inputs)
    >>> features = compute_feat(spectr)
    >>> features.shape
    ... torch.Size([10, 101, 1])
    """

    def __init__(
        self,
        sample_rate,
        win_length=25,
        hop_length=10,
        n_fft=400,
        window_fn=torch.hamming_window,
        m=10,
        r=30,
        freq_range=None,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.window = window_fn(win_length)
        self.freq_range = freq_range
        self.eps = 1e-5
        self.m = m
        self.r = r
        self.stft = STFT(
            sample_rate=sample_rate,
            win_length=win_length,
            hop_length=hop_length,
            n_fft=n_fft,
            window_fn=window_fn,
        )

    def _compute_welch_spectrum(self, stft):
        n = stft.shape[1]
        welch_spectr = torch.zeros_like(stft)

        stft = stft / self.sample_rate * (self.window**2).sum()
        stft[..., 1:-1] *= 2

        for i in range(1, self.m + 1):
            welch_spectr[:, i, :] = stft[:, :i, :].mean(dim=1)
        for i in range(self.m + 1, n):
            welch_spectr[:, i, :] = stft[:, i-self.m:i, :].mean(dim=1)

        return welch_spectr

    def _compute_geometric_mean(self, stft):
        m = stft.shape[1]
        geometric_mean = torch.zeros_like(stft)

        for i in range(1, self.r + 1):
            geometric_mean[:, i, :] = (
                (stft[:, :i, :] + self.eps).log().mean(dim=1).exp()
            ) - self.eps
        for i in range(self.r + 1, m):
            geometric_mean[:, i, :] = (
                (stft[:, i-self.r:i, :] + self.eps).log().mean(dim=1).exp()
            ) - self.eps

        return geometric_mean

    def _compute_arithmetic_mean(self, stft):
        m = stft.shape[1]
        arithmetic_mean = torch.zeros_like(stft)

        for i in range(1, self.r + 1):
            arithmetic_mean[:, i, :] = stft[:, :i, :].mean(dim=1)
        for i in range(self.r + 1, m):
            arithmetic_mean[:, i, :] = stft[:, i-self.r:i, :].mean(dim=1)

        return arithmetic_mean

    def forward(self, x):
        """
        x : tensor
            The name-value arguments apply if x is a batch of time-domain
            signals to transform. The tensor must have the format
            (batch, time_step). If x is a batch of frequency-domain signal,
            name-value arguments are ignored. The tensor must ave the format
            (batch, time_step, n_fft/2 + 1, 2).
        """

        if x.ndim == 2:
            x = self.stft(x)

        spectr = spectral_magnitude(x) / self.m

        if self.freq_range is not None:
            n_fft = (spectr.shape[2] - 1) * 2
            bin_s = int(self.freq_range[0] / self.sample_rate * n_fft)
            bin_e = int(self.freq_range[1] / self.sample_rate * n_fft)
            spectr = spectr[..., bin_s:bin_e]

        spectr = self._compute_welch_spectrum(spectr)
        gm = self._compute_geometric_mean(spectr) + self.eps
        am = self._compute_arithmetic_mean(spectr) + self.eps

        flatness = -1.0 * (gm / am).log10().sum(dim=-1, keepdim=True)

        return flatness
