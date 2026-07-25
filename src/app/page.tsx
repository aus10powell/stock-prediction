import { StockForecastApp } from "@/components/StockForecastApp";
import { runAnalysis } from "@/lib/analysis";
import { parseSettings } from "@/lib/settings";

export default async function Home(props: PageProps<"/">) {
  const settings = parseSettings(await props.searchParams);
  const analysis = await runAnalysis(settings);

  return <StockForecastApp settings={settings} analysis={analysis} />;
}
