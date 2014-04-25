package edu.buct.glasearch.search.data;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.nio.charset.Charset;

import org.apache.commons.compress.utils.IOUtils;
import org.apache.commons.lang.StringUtils;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.hbase.HBaseConfiguration;
import org.apache.hadoop.hbase.client.HTable;
import org.apache.hadoop.hbase.client.Put;
import org.apache.hadoop.hbase.util.Bytes;

import com.mycila.xmltool.XMLDoc;
import com.mycila.xmltool.XMLDocumentException;
import com.mycila.xmltool.XMLTag;

import edu.buct.glasearch.search.jobs.ImageSearchJob;

public class ImportAnnotationData {
	
	private static final String FILE_ENCODING = "ISO-8859-1";

	private static Log logger = LogFactory.getLog(ImportAnnotationData.class);
	
	private static final String HBASE_LOCATION = "hdfs://cluster1.centos:8020";
	private static final String dataLocation = "/root/development/data/annotations_complete_eng";
	
	public void setFileEncoding() throws IOException {
		
		String encodingHead = "<?xml version=\"1.0\" encoding=\"ISO-8859-1\"?>\n";
		
        File dir = new File(dataLocation);
        
        int counter = 0;
        for (File subDir : dir.listFiles()) {
        	for (File image : subDir.listFiles()) {
        		InputStream is = new FileInputStream(image);
        		BufferedReader reader = new BufferedReader(new InputStreamReader(is, Charset.forName(FILE_ENCODING)));
        		
        		String line = reader.readLine();
        		
        		if (StringUtils.isEmpty(line) || line.contains(FILE_ENCODING)) continue;

        		BufferedWriter writer = new BufferedWriter(new FileWriter(image));
        		writer.write(encodingHead.toCharArray());
        		while (line != null) {
        			writer.write(line);
        			writer.write('\n');
        			line = reader.readLine();
        		}
        		is.close();
        		writer.close();
        		counter++;
        	}
        	logger.info("procesed: " + counter);
        }
	}
	
	public void doSomething() throws IOException {
    	
        Configuration conf = HBaseConfiguration.create();
	    conf.set("fs.defaultFS", HBASE_LOCATION);
	    
        HTable table = new HTable(conf, ImageSearchJob.imageInfoTable);
        
        File dir = new File(dataLocation);
        
        int errorCounter = 0;
        for (File subDir : dir.listFiles()) {
        	for (File image : subDir.listFiles()) {
        		
        		String imageId = image.getName().substring(0, image.getName().indexOf('.'));
        		String rowId = subDir.getName() + "-" + imageId;
        		
                Put put = new Put(Bytes.toBytes(rowId));
                put.add(ImageSearchJob.COLUMN_FAMILY_BYTES, 
                		Bytes.toBytes("category"), Bytes.toBytes(subDir.getName()));
                put.add(ImageSearchJob.COLUMN_FAMILY_BYTES, 
                		Bytes.toBytes("imageid"), Bytes.toBytes(imageId));
                
                XMLTag doc = null;
                try {
                	doc = XMLDoc.from(image);
                } catch (XMLDocumentException e) {
                	logger.error("Error opening:" + rowId, e);
                	errorCounter++;
                	continue;
                }
        		for (XMLTag tag : doc.getChilds()) {
        			String tagName = tag.getCurrentTagName().toLowerCase();
        			String value = tag.getText();
        			put.add(ImageSearchJob.COLUMN_FAMILY_BYTES, Bytes.toBytes(tagName), Bytes.toBytes(value));
        		}
                table.put(put);
        	}
            table.flushCommits();
            logger.warn("error count:" + errorCounter);
        }
        
        table.close();
		
	}
	
    public static void main(String[] args) throws Exception {
    	ImportAnnotationData importData = new ImportAnnotationData();
    	//importData.setFileEncoding();
    	importData.doSomething();
    }
}
