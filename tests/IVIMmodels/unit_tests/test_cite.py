from src.wrappers.OsipiBase import OsipiBase


def test_cite_returns_text(capsys):
    """cite() should return the citation text, and only print it when verbose=True."""
    fit = OsipiBase(algorithm="IAR_LU_biexp", bvalues=[0, 50, 200, 800])
    capsys.readouterr()

    text = fit.cite(verbose=False)
    assert isinstance(text, str)
    assert "IAR_LU_biexp" in text
    assert "0, 50, 200, 800 s/mm" in text
    assert capsys.readouterr().out == ""

    assert fit.cite() == text
    assert text in capsys.readouterr().out
