using EGA.SPC.BLL.Model;
using EGA.SPC.Domain;
using EGA.SPC.Dto;
using EGA.SPC.Dto.Calculus;
using System;
using System.Collections.Generic;
using System.Linq;

namespace EGA.SPC.BLL
{
	public partial class ChartBase
	{
		protected bool Between(double num, double lower, double upper, bool inclusive = true)
		{
			return inclusive
				? lower <= num && num <= upper
				: lower < num && num < upper;
		}

		protected DtoSigmasStatistics CalculateSigmasStatistics(CalculusParameter param, double setPoint, double OneSigma, List<double?> sampleValues)
		{
			DtoSigmasStatistics result = new DtoSigmasStatistics();

			List<double> doubleValues = sampleValues.Where(x => x.HasValue).Select(x => (double)x.Value).ToList();
			result.WithinOneSigma.TotalOfElements = doubleValues.Where(x => Between(x, setPoint - OneSigma, setPoint + OneSigma)).Count();
			result.WithinTwoSigmas.TotalOfElements = doubleValues.Where(x => Between(x, setPoint - (OneSigma * 2), setPoint + (OneSigma * 2))).Count();
			result.WithinThreeSigmas.TotalOfElements = doubleValues.Where(x => Between(x, setPoint - (OneSigma * 3), setPoint + (OneSigma * 3))).Count();
			result.AboveLSE.TotalOfElements = doubleValues.Where(x => x > param.variable.LSE).Count();
			result.BellowLIE.TotalOfElements = doubleValues.Where(x => x < param.variable.LIE).Count();
			result.AboveThreeSigmas.TotalOfElements = doubleValues.Where(x => x > setPoint + (OneSigma * 3)).Count();
			result.BellowThreeSigmas.TotalOfElements = doubleValues.Where(x => x < setPoint - (OneSigma * 3)).Count();

			calculateSigmaPerc(result.WithinOneSigma, sampleValues.Count);
			calculateSigmaPerc(result.WithinTwoSigmas, sampleValues.Count);
			calculateSigmaPerc(result.WithinThreeSigmas, sampleValues.Count);
			calculateSigmaPerc(result.AboveLSE, sampleValues.Count);
			calculateSigmaPerc(result.BellowLIE, sampleValues.Count);
			calculateSigmaPerc(result.AboveThreeSigmas, sampleValues.Count);
			calculateSigmaPerc(result.BellowThreeSigmas, sampleValues.Count);

			result.SampleSize = sampleValues.Count;

			return result;
		}

		public double CalculateAmplitude(List<double?> samples)
		{
			double min = samples.Min(i => i ?? 0);
			double max = samples.Max(i => i ?? 0);
			double result = max - min;
			return result;
		}

		public double calculateFx(double average, double stdDerivation, double x)
		{
			if (stdDerivation == 0) return 0;

			//var a = 1.0 / Math.Sqrt(2 * Math.PI * stdDerivation);
			var a = 1.0 / (stdDerivation * Math.Sqrt(2.0 * Math.PI));
			var b = Math.Exp(-0.5 * Math.Pow((x - average) / stdDerivation, 2));
			return a * b;
		}

		protected void calculateSigmaPerc(DtoSigmaStatistics item, int sampleSize)
		{
			if (item.TotalOfElements == 0) return;
			item.Percentage = ((double)item.TotalOfElements / (double)sampleSize) * 100.0;
		}


		public double CalculateStandardDeviation(List<double?> samples)
		{
			List<double> doubleSamples = samples.Select(i => i ?? 0).ToList();

			double variance = CalculateVariance(doubleSamples);

			if (variance == 0) return 0;

			double result = Math.Sqrt(variance);

			return result;
		}

		public double CalculateVariance(List<double> samples)
		{
			double S2 = 0.0;

			// Sample = N-1
			int totalElements = samples.Count - 1;

			if (totalElements == 0) return 0;

			double average = samples.Average();

			if (average == 0) return 0;

			foreach (var value in samples)
			{
				double difference = (value - average);
				double pow = Math.Pow(difference, 2);
				double x = pow / totalElements;
				S2 += x;
			}

			return S2;
		}

		public DtoInMetro248Results CalculateInMetro248(int SubGroupSize, List<double?> Samples, double X, double Qn, double S)
		{
			List<double> doubleSamples = Samples.Select(i => i ?? 0).ToList();

			DtoInMetro248Results results = new DtoInMetro248Results();
			results.Qn = Qn;
			results.StandardDerivation = S;

			// Critério da media

			//int sampleSize = Samples.Count;
			//results.SampleSize = sampleSize;

			switch (SubGroupSize)
			{
				case 5:
					results.Table2c = 0;
					results.Factor = 2.059;
					break;
				case 13:
					results.Table2c = 1;
					results.Factor = 0.847;
					break;
				case 20:
					results.Table2c = 1;
					results.Factor = 0.640;
					break;
				case 32:
					results.Table2c = 2;
					results.Factor = 0.485;
					break;
				case 80:
					results.Table2c = 5;
					results.Factor = 0.295;
					break;
				default:
					results.SubGroupSizeNotFoundInCriteria = true;
					return results;
			}

			results.Criterion = string.Format("X >= Qn - ({0} * S)", results.Factor);

			double compair = Qn - (results.Factor * S);
			results.ApprovedAtAverageCriteria = X >= compair;

			// Critério individual

			if (Qn < 50)
				results.Table1ToleranceInPercentage = 9;
			else
				if (Qn < 100)
				results.Table1ToleranceInQty = 4.5;
			else
					if (Qn < 200)
				results.Table1ToleranceInPercentage = 4.5;
			else
						if (Qn < 300)
				results.Table1ToleranceInQty = 9;
			else
							if (Qn < 500)
				results.Table1ToleranceInPercentage = 3;
			else
								if (Qn < 1000)
				results.Table1ToleranceInQty = 15;
			else
									if (Qn < 10000)
				results.Table1ToleranceInPercentage = 1.5;
			else
										if (Qn < 15000)
				results.Table1ToleranceInQty = 150;
			else
				results.Table1ToleranceInPercentage = 1;

			if (results.Table1ToleranceInPercentage > 0)
			{
				double percentage = (Qn * results.Table1ToleranceInPercentage) / 100.0;
				results.SetPointForTableOne = Qn - percentage;
			}

			if (results.Table1ToleranceInQty > 0)
				results.SetPointForTableOne = Qn - results.Table1ToleranceInQty;

			List<double> samplesBellow = doubleSamples.Where(x => x < results.SetPointForTableOne).ToList();

			results.TotalElementsBelowC = samplesBellow.Count;

			if (results.TotalElementsBelowC == 0) results.ApprovedAtIndividualCriteria = true;
			else
				results.ApprovedAtIndividualCriteria = results.TotalElementsBelowC < results.Table2c;

			results.Approved = results.ApprovedAtAverageCriteria && results.ApprovedAtIndividualCriteria;

			return results;
		}

		// Remove itens do histograma que estiverem com valores zero antes e depois do LIE E LSE

		protected void ClearHistogramOutbounds(DtoHistogram Histogram, CalculusParameter param)
		{
			if (!param.removeHistogramOutbounds) return;

			if (Histogram.Items.Count < 1) return;

			int firstValueIndex = Histogram.Items.FindIndex(x => x.Max > param.variable.LIE && x.Frequency > 0);
			if (firstValueIndex > 1) Histogram.Items = Histogram.Items.Where((x, index) => index >= firstValueIndex - 1).ToList();

			int lastValueIndex = Histogram.Items.FindLastIndex(x => x.Min >= param.variable.LSE && x.Frequency > 0);

			if (lastValueIndex < 0)
				lastValueIndex = Histogram.Items.FindIndex(x => x.Max > param.variable.LSE && x.Frequency == 0);

			if (lastValueIndex > 0) Histogram.Items = Histogram.Items.Take(lastValueIndex + 1).ToList();

			// Insere primeiro e ultimo bloco com frequencia 0

			DtoHistogramItem first = (DtoHistogramItem)Histogram.Items.First().Clone();
			double incr = first.Max - first.Min;

			if (firstValueIndex < 0)
			{
				first.Frequency = 0;
				first.Min -= incr;
				first.Max -= incr;
				Histogram.Items.Insert(0, first);
			}

			if (lastValueIndex < 0)
			{
				DtoHistogramItem last = (DtoHistogramItem)Histogram.Items.Last().Clone();
				last.Frequency = 0;
				last.Min += incr;
				last.Max += incr;
				Histogram.Items.Add(last);
			}
		}

		// Curva suavizada
		protected DtoBellCurve CreateBellCurve(DtoHistogram histogram, double average, double standardDeviationWithin, double standardDeviationOverall)
		{
			DtoBellCurve result = new DtoBellCurve();
			DtoHistogramItem lastItem = null;

			foreach (DtoHistogramItem item in histogram.Items)
			{
				if (lastItem == null)
				{
					DtoBellCurveItem newItem = new DtoBellCurveItem();
					newItem.X = item.Center;
					newItem.WithinCurve = calculateFx(average, standardDeviationWithin, item.Center);
					newItem.OverallCurve = calculateFx(average, standardDeviationOverall, item.Center);
					result.Items.Add(newItem);
				}
				else
				{
					double range = item.Center - lastItem.Center;
					int divisions = 10;
					double incr = range / (double)divisions;
					double x = lastItem.Center;

					while (divisions > 0)
					{
						x += incr;
						DtoBellCurveItem newItem = new DtoBellCurveItem();
						newItem.X = x;
						newItem.WithinCurve = calculateFx(average, standardDeviationWithin, x);
						newItem.OverallCurve = calculateFx(average, standardDeviationOverall, x);
						result.Items.Add(newItem);
						--divisions;
					}
				}

				lastItem = item;
			}

			return result;
		}

		// K-> Number of classes
		protected DtoHistogram CreateHistogram(List<double> values, ChartHistogramType histogramType = ChartHistogramType.Sqrt, int k = 0, int precision = 0)
		{
			try
			{
				DtoHistogram result = new DtoHistogram();
				var orderedValues = values.OrderBy(x => x).ToList();
				double start = 0;

				if (histogramType == ChartHistogramType.Distinct)
				{
					var samples = from n in orderedValues
								  group n by n;

					foreach (var g in samples)
					{
						start = g.Key;
						DtoHistogramItem newItem = new DtoHistogramItem() { Min = start, Max = start, Frequency = g.Count() };
						result.Items.Add(newItem);
					}

					return result;
				}

				List<double> intervals = new List<double> { 0.00001, 0.0001, 0.001, 0.01, 0.1, 0, 1, 2, 5, 10, 20, 25, 50, 100, 500, 1000, 5000, 10000 };

				switch (histogramType)
				{
					case ChartHistogramType.Sqrt:
						k = (int)Math.Sqrt(values.Count);
						break;
					case ChartHistogramType.Sturge:
						k = (int)(1.0 + Math.Log(values.Count, 2));
						break;
					case ChartHistogramType.Distinct:
						k = (int)values.Distinct().Count();
						break;
					case ChartHistogramType.Custom:
						break;
				}

				if (orderedValues.Count > 0)
				{
					result.MinValue = orderedValues.Min();
					result.MaxValue = orderedValues.Max();
					result.R = result.MaxValue - result.MinValue;
					result.K = k;
					double i = result.R / result.K;

					if (histogramType == ChartHistogramType.Distinct)
						i = Math.Round(i, precision);

					double fi = 0;
					if (intervals.Exists(x => x >= i))
						fi = intervals.FirstOrDefault(x => x >= i);
					else
						fi = intervals.Max();

					double center = result.MinValue + (result.R / 2);

					start = center - (fi * (k / 2.0));

					double end = start + fi;

					bool hasItemsAtFirstCell = orderedValues.Exists(x => x >= start && x < end);

					if (hasItemsAtFirstCell)
					{
						start -= fi;
						k += 2;
					}

					for (int y = 0; y <= k; y++)
					{
						DtoHistogramItem newItem = new DtoHistogramItem();
						newItem.Min = start;
						newItem.Max = start + fi;
						newItem.Frequency = orderedValues.Count(x => x >= newItem.Min && x < newItem.Max);
						start += fi;
						result.Items.Add(newItem);
					}
				}

				return result;
			}
			catch (Exception ex)
			{
				throw;
			}
		}

		protected TrafficLight GetTrafficLightFromCPK(double CPK, ChartControlVariable variable)
		{
			TrafficLight result = TrafficLight.NotCapable;

			if (CPK > variable.CPKHigh) result = TrafficLight.Excellent;
			else
				if (CPK > variable.CPKNormal) result = TrafficLight.Capable;
			else
					if (CPK > variable.CPKLow) result = TrafficLight.Acceptable;

			return result;
		}

		protected TrafficLight GetTrafficLightFromPPK(double PPK, ChartControlVariable variable)
		{
			TrafficLight result = TrafficLight.NotCapable;

			if (PPK > variable.PPKHigh) result = TrafficLight.Excellent;
			else
				if (PPK > variable.PPKNormal) result = TrafficLight.Capable;
			else
					if (PPK > variable.PPKLow) result = TrafficLight.Acceptable;

			return result;
		}

		protected double GetZForLIE(double average, double stdDerivation, double x)
		{
			double result = 0;

			if (stdDerivation != 0)
				result = (average - x) / stdDerivation;

			return result;
		}

		protected double GetZForLSE(double average, double stdDerivation, double x)
		{
			double result = 0;

			if (stdDerivation != 0)
				result = (x - average) / stdDerivation;

			return result;
		}
	}
}