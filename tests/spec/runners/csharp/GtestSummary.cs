// The run's skipped and expected-failure counts, written once after the spec tests.

namespace SpecRunner;

/// <summary>
/// The class fixture of <see cref="SpecTests"/> that writes the <c>gtest summary</c> line when
/// the run's cases are done: the cases skipped by <c>skip:</c> and the ones that failed as
/// expected.
/// </summary>
public sealed class GtestSummary : IDisposable
{
    /// <summary>Cases skipped by <c>skip:</c>.</summary>
    public static int Skipped;

    /// <summary>Cases marked <c>xfail:</c> that failed as expected.</summary>
    public static int Xfailed;

    public void Dispose() => Console.WriteLine($"gtest summary: skipped={Skipped} xfailed={Xfailed}");
}
