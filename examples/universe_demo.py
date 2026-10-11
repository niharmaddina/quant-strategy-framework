"""
Demonstration of universe construction.
Shows how to build a sophisticated trading universe.
"""

import os
import sys

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from universe.constructor import UniverseConstructor


def demo_basic_universe():
    """Demonstrate frozen/manual universe usage (recommended for backtests)."""
    print("\n" + "="*70)
    print("DEMO 1: Basic Universe Construction")
    print("="*70)
    
    # Load the built-in frozen universe file (no yfinance metadata calls)
    universe_file = os.path.join(project_root, "universe", "universe_mean_reversion_manual_50.txt")
    symbols = UniverseConstructor.load_universe(universe_file)
    universe = UniverseConstructor(symbols=symbols)
    
    print(f"\nFrozen universe size: {len(symbols)} symbols")
    print(f"Symbols: {', '.join(symbols[:15])} ...")
    
    # Local sector tagging (no API)
    sector_counts = {}
    for s in symbols:
        sector = universe.get_symbol_sector(s) or "N/A"
        sector_counts[sector] = sector_counts.get(sector, 0) + 1
    
    print("\nSector distribution (from curated mapping):")
    for sector, count in sorted(sector_counts.items(), key=lambda x: (-x[1], x[0])):
        print(f"  {sector:25s}: {count:2d}")


def demo_filtering():
    """Demonstrate deterministic sector-balanced selection without metadata calls."""
    print("\n" + "="*70)
    print("DEMO 2: Advanced Filtering")
    print("="*70)
    
    universe = UniverseConstructor()
    selected = universe.create_manual_sector_balanced_universe(
        target_size=25,
        stocks_per_sector=2,
        include_index_etfs=True
    )
    print(f"\nManual sector-balanced universe: {len(selected)} symbols")
    print(f"Symbols: {', '.join(selected)}")


def demo_sector_balancing():
    """Demonstrate sector-balanced universe construction."""
    print("\n" + "="*70)
    print("DEMO 3: Sector-Balanced Universe")
    print("="*70)
    
    universe = UniverseConstructor()
    
    # Create balanced universe
    balanced = universe.create_sector_balanced_universe(
        target_size=20,
        stocks_per_sector=2
    )
    
    print(f"\nSector-balanced universe: {len(balanced)} symbols")
    
    # Show sector distribution
    sector_counts = {}
    for symbol in balanced:
        sector = universe.get_symbol_sector(symbol)
        sector_counts[sector] = sector_counts.get(sector, 0) + 1
    
    print("\nSector Distribution:")
    for sector, count in sorted(sector_counts.items()):
        print(f"  {sector:25s}: {count:2d} stocks")
    
    # Analyze exposure
    print("\n" + "-"*70)
    print("Sector Exposure Analysis:")
    exposure = universe.analyze_sector_exposure()
    print(exposure.to_string(index=False))


def demo_correlation_analysis():
    """Correlation/scoring/optimized universes require yfinance metadata/returns and are omitted here."""
    print("\n" + "="*70)
    print("DEMO 4: Correlation/Scoring/Optimized Universes (Skipped)")
    print("="*70)
    print("\nThese features are rate-limit sensitive and require network access.")
    print("For historical backtests, prefer frozen/manual universes.")
    return


#
# The previous version of this demo included scoring/optimized universes and correlation analysis.
# Those require yfinance metadata/returns and frequently hit rate limits; we keep this demo offline-friendly.


def demo_export_import():
    """Demonstrate exporting and importing universes."""
    print("\n" + "="*70)
    print("DEMO 5: Export & Import")
    print("="*70)
    
    universe = UniverseConstructor()
    balanced = universe.create_manual_sector_balanced_universe(
        target_size=15,
        stocks_per_sector=1,
        include_index_etfs=True
    )
    universe.symbols = balanced
    
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        txt_file = os.path.join(tmpdir, 'universe.txt')
        print("\nExporting universe to temp file...")
        universe.export_universe(txt_file, include_metadata=False)
        print("  ✓ Exported")
    
    # Import
    print("\nImporting universe from file...")
    imported = UniverseConstructor.load_universe(txt_file)
    print(f"  ✓ Imported {len(imported)} symbols")
    print(f"  Symbols: {', '.join(imported[:10])}...")


def main():
    """Run all demonstrations."""
    print("\n" + "="*70)
    print("UNIVERSE CONSTRUCTION DEMONSTRATION")
    print("="*70)
    print("\nThis demo shows how to build a sophisticated trading universe.")
    
    try:
        # Run all demos
        demo_basic_universe()
        demo_filtering()
        demo_sector_balancing()
        demo_correlation_analysis()
        demo_export_import()
        
        print("\n" + "="*70)
        print("ALL DEMONSTRATIONS COMPLETE!")
        print("="*70)
        print("\nKey Takeaways:")
        print("  1. Prefer frozen/manual universes for historical backtests")
        print("  2. Sector tagging and manual sector-balance is deterministic (no API calls)")
        print("  3. yfinance metadata/returns features are rate-limit sensitive")
        print("\n" + "="*70 + "\n")
        
    except Exception as e:
        print(f"\n⚠ Error during demonstration: {e}")
        print("This may be due to network issues or API rate limits.")
        print("The code is still functional - try again later.\n")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

