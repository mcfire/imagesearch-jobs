package edu.buct.glasearch.search.jobs;

import java.io.IOException;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;

import net.semanticmetadata.lire.imageanalysis.EdgeHistogram;
import net.semanticmetadata.lire.imageanalysis.LireFeature;
import net.semanticmetadata.lire.imageanalysis.SimpleColorHistogram;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.hbase.client.HTable;
import org.apache.hadoop.hbase.client.Put;
import org.apache.hadoop.hbase.client.Result;
import org.apache.hadoop.hbase.client.ResultScanner;
import org.apache.hadoop.hbase.client.Scan;
import org.apache.hadoop.hbase.filter.RandomRowFilter;
import org.apache.hadoop.hbase.io.ImmutableBytesWritable;
import org.apache.hadoop.hbase.mapreduce.TableMapReduceUtil;
import org.apache.hadoop.hbase.mapreduce.TableMapper;
import org.apache.hadoop.hbase.mapreduce.TableReducer;
import org.apache.hadoop.hbase.util.Bytes;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapreduce.Job;

import com.google.gson.Gson;

public class ImageDistancePrepareJob {

	private static final Log logger = LogFactory.getLog(ImageDistancePrepareJob.class);
	
	private static int CALCULATE_DISTANCE_COUNT = 3;

	public static class Map extends TableMapper<Text, DoubleWritable> {
		
		int rowCount = 20;
	
		@Override
		protected void setup(Context context) throws IOException,
				InterruptedException {
			Configuration config = context.getConfiguration();
			rowCount = config.getInt("rowCount", rowCount);
		}
		
		@Override
		protected void map(ImmutableBytesWritable rowKey, Result result,
				Context context) throws IOException, InterruptedException {
			
			if (logger.isInfoEnabled()) {
				logger.info("calculate distance, owner:" + new String(rowKey.get()));
			}
			
			LireFeature colorFeature = rowToColorFeature(result);
			LireFeature edgeFeature = rowToEdgeFeature(result);
			if (colorFeature == null || edgeFeature == null) return;
			
			HTable imageTable = new HTable(context.getConfiguration(), ImageSearchJob.imageInfoTable);
			Scan randomScan = new Scan();
			randomScan.setFilter(new RandomRowFilter((float)(CALCULATE_DISTANCE_COUNT * 1.2 / rowCount)));
			ResultScanner rs = imageTable.getScanner(randomScan);
			
			int i = 0;
			for (Result r = rs.next(); r != null && i < CALCULATE_DISTANCE_COUNT; r = rs.next(), i++) {
				
				if (Bytes.equals(rowKey.get(), r.getRow())) {
					i--;
					continue;
				}
				
				LireFeature targetColorFeature = rowToColorFeature(r);
				LireFeature targetEdgeFeature = rowToEdgeFeature(r);
				if (targetColorFeature == null || targetEdgeFeature == null) continue;
				
				double colorDistance = colorFeature.getDistance(targetColorFeature);
				double edgeDistance = edgeFeature.getDistance(targetEdgeFeature);
				
				if (logger.isInfoEnabled()) {
					logger.info("calculate distance, owner:" + new String(rowKey.get()) + 
							", target: " + new String(r.getRow()) + 
							",color distance:" + colorDistance + 
							",edge distance:" + edgeDistance);
				}
				
				context.write(new Text(ImageSearchJob.COLOR_FEATURE_RESULT), 
						new DoubleWritable(colorDistance));
				context.write(new Text(ImageSearchJob.EDGE_FEATURE_RESULT), 
						new DoubleWritable(edgeDistance));
			}
			imageTable.close();
		}

		private LireFeature rowToColorFeature(Result result) {
			byte[] featureBytes = result.getValue(ImageSearchJob.COLUMN_FAMILY_BYTES, 
					ImageSearchJob.COLOR_FEATURE_COLUMN);
			if (featureBytes == null) return null;
			
			LireFeature feature = new SimpleColorHistogram();
			feature.setByteArrayRepresentation(featureBytes);
			return feature;
		}

		private LireFeature rowToEdgeFeature(Result result) {
			byte[] featureBytes = result.getValue(ImageSearchJob.COLUMN_FAMILY_BYTES, 
					ImageSearchJob.EDGE_FEATURE_COLUMN);
			if (featureBytes == null) return null;
			
			LireFeature feature = new EdgeHistogram();
			feature.setByteArrayRepresentation(featureBytes);
			return feature;
		}
	}

	public static class Reduce extends TableReducer<Text, DoubleWritable, ImmutableBytesWritable> {

		@Override
		protected void reduce(Text rowKey, Iterable<DoubleWritable> values,
				Context context) throws IOException, InterruptedException {
			
			List<Double> valueList = new LinkedList<Double>();
			
			int n = 0;
			double avg = 0d, sigma = 0d;
			Iterator<DoubleWritable> itor = values.iterator();
			
			while (itor.hasNext()) {
				double v = itor.next().get();
				valueList.add(v);
				n++;
				
				avg = ((n-1) * avg) / n + v / n;
			}
			
			int i = 0;
			for (Double v : valueList) {
				double x = Math.pow(v - avg, 2);
				i++;
				
				sigma = ((i-1) * sigma) / i + x / i;
			}
			sigma = Math.sqrt(sigma);
			
			if (logger.isInfoEnabled()) {
				logger.info("calculate group info: " + rowKey.toString() + 
						". n:" + n + ",avg:" + avg +
						",sigma:" + sigma);
			}
			
			Put put = new Put(Bytes.toBytes(rowKey.toString()));
			put.add(ImageSearchJob.COLUMN_FAMILY_BYTES, Bytes.toBytes("count"), Bytes.toBytes(n));
			put.add(ImageSearchJob.COLUMN_FAMILY_BYTES, Bytes.toBytes("avg"), Bytes.toBytes(avg));
			put.add(ImageSearchJob.COLUMN_FAMILY_BYTES, Bytes.toBytes("sigma"), Bytes.toBytes(sigma));
			context.write(null, put);
		}
	}
	
	/**
	 * Job configuration.
	 */
	public static Job configureJob(Configuration conf, byte[] startRow, byte[] stopRow)
			throws IOException {

		//important: use this method to add job and it's dependency jar
		TableMapReduceUtil.addDependencyJars(conf, ImageDistancePrepareJob.class, LireFeature.class, Gson.class);
		
		JobConf jobConf = new JobConf(conf);
		jobConf.setJobName("image-distance-caculate");
		
		Job job = new Job(jobConf);
		job.setJarByClass(ImageDistancePrepareJob.class);
		
		Scan scan = new Scan();
		scan.setCaching(500);        // 1 is the default in Scan, which will be bad for MapReduce jobs
		scan.setCacheBlocks(false);  // don't set to true for MR jobs
		scan.setStartRow(startRow);
		scan.setStopRow(stopRow);
		// set other scan attrs
		
		TableMapReduceUtil.initTableMapperJob(
				ImageSearchJob.imageInfoTable,        // input table
				scan,               // Scan instance to control CF and attribute selection
				Map.class,     // mapper class
				Text.class,         // mapper output key
				DoubleWritable.class,  // mapper output value
				job);
		
		TableMapReduceUtil.initTableReducerJob(
				ImageSearchJob.imageDistanceTable,      // output table
				Reduce.class,             // reducer class
				job);
		job.setNumReduceTasks(1);
		
		return job;
	}

	public static void main(String[] args) throws Exception {
		Configuration conf = new Configuration();
	    //conf.set("fs.defaultFS", "hdfs://cluster1.centos:8020");
	    //conf.set("yarn.resourcemanager.address", "cluster1.centos:8032");
	    //conf.set("mapreduce.framework.name", "yarn");
		String START_ROW = "39";
		String STOP_ROW = "39-399990";

		HTable imageTable = new HTable(conf, ImageSearchJob.imageInfoTable);
		ResultScanner scanner = imageTable.getScanner(ImageSearchJob.COLUMN_FAMILY_BYTES);
		int rowCount = 0;
		while (scanner.next() != null) {
			rowCount++;
		}
		imageTable.close();
		
		conf.setInt("rowCount", rowCount);
	    
		Job job = configureJob(conf, Bytes.toBytes(START_ROW),  Bytes.toBytes(STOP_ROW));

		boolean isSuccess = job.waitForCompletion(true);
		
		if (isSuccess) {
			
		}
	    
		System.exit(isSuccess ? 0 : -1);
	}
}
